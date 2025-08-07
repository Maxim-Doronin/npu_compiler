//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {

namespace detail {
std::optional<BatchCompilerOptionsAdapterView::Occurence> extractSubOption(StrOption& subOption,
                                                                           mlir::detail::PassOptions& passOptions,
                                                                           std::string_view strOptions) {
    auto namePos = strOptions.find(subOption.getArgStr());
    if (namePos == std::string::npos) {
        return {};
    }

    static const std::string unexpectedOptionValue{"unexpected-option-value"};
    auto strOptionsBeginIt = strOptions.begin();
    std::advance(strOptionsBeginIt, namePos);
    subOption = unexpectedOptionValue;
    auto statusCode = passOptions.parseFromString(
            std::string_view(&(*strOptionsBeginIt), static_cast<size_t>(strOptions.end() - strOptionsBeginIt)));
    (void)statusCode;
    if (subOption == unexpectedOptionValue) {
        return {};
    }
    if (subOption.empty()) {
        return std::make_optional<BatchCompilerOptionsAdapterView::Occurence>(
                BatchCompilerOptionsAdapterView::Occurence{namePos, subOption.getArgStr().size() + 1});
    }

    auto valuePos = strOptions.find(subOption, namePos);
    if (valuePos == std::string::npos) {
        return {};
    }

    // Include leading '{' and trailing '}' if it exists
    auto subOptionSize = subOption.size();
    if (valuePos > 0 && strOptions[valuePos - 1] == '{') {
        valuePos--;
        subOptionSize++;
    }
    if ((valuePos + subOptionSize) < strOptions.size() && strOptions[valuePos + subOptionSize] == '}') {
        subOptionSize++;
    }

    return std::make_optional<BatchCompilerOptionsAdapterView::Occurence>(
            BatchCompilerOptionsAdapterView::Occurence{namePos, valuePos + subOptionSize - namePos});
}

std::string cutOccurenceOutOfString(const std::optional<BatchCompilerOptionsAdapterView::Occurence>& occurence,
                                    const std::string& originalStrOptions) {
    if (!occurence.has_value()) {
        return originalStrOptions;
    }
    std::string ret = originalStrOptions.substr(0, occurence->pos);
    VPUX_THROW_WHEN(occurence->pos + occurence->length > originalStrOptions.size(),
                    "in cutOccurenceOutOfString cannot cut string from pos: {0}, which is bigger than an entire string "
                    "length: {1}",
                    occurence->pos, originalStrOptions.size());
    ret += originalStrOptions.substr(occurence->pos + occurence->length);
    return ret;
}
}  // namespace detail

BatchCompileOptionsAdapter::BatchCompileOptionsAdapter(mlir::detail::PassOptions& parent)
        : batchCompileMethod(parent, "batch-compile-method",
                             llvm::cl::desc("Preferred method for compilation of batched networks. Supported methods: "
                                            "\"unroll\", \"debatch\". Default is \"debatch\""),
                             llvm::cl::init("unroll")),
          debatchCompileMethodSettings(
                  parent, "debatcher-settings",
                  llvm::cl::desc("Additional parameters, applied when \"batch-compile-method\" is \"debatch\"."),
                  llvm::cl::init(getDefaultValueOfStrSubOption<DebatcherOptions>())),
          batchUnrollCompileMethodSettings(
                  parent, "batch-unroll-settings",
                  llvm::cl::desc("Additional parameters, applied when \"batch-compile-method\" is \"unroll\"."),
                  llvm::cl::init(getDefaultValueOfStrSubOption<BatchUnrollOptions>())) {
}

void BatchCompileOptionsAdapter::updateBatchCompileOptionsFromString(std::string_view strOptions) {
    auto view = BatchCompilerOptionsAdapterView::tryExtractFromString(strOptions);
    if (view.has_value()) {
        *this = view->get();
    }
}

std::optional<BatchCompilerOptionsAdapterView> BatchCompilerOptionsAdapterView::tryExtractFromString(
        std::string_view strOptions) {
    if (strOptions.empty()) {
        return {};
    }

    // Do not use mlir::detail::PipelineOptions::createFromString() as it will return error,
    // when faces unrecognizable params in original string
    auto passOptions = std::make_unique<mlir::detail::PassOptions>();
    auto extractedOptions = std::make_unique<BatchCompileOptionsAdapter>(*passOptions);

    BatchCompilerOptionsAdapterView view;
    extractedOptions->batchCompileMethod = "";  // clear an option value including default
    view.optionDataMemberViews.emplace_back(
            detail::extractSubOption(extractedOptions->batchCompileMethod, *passOptions, strOptions));
    view.optionDataMemberViews.emplace_back(
            detail::extractSubOption(extractedOptions->debatchCompileMethodSettings, *passOptions, strOptions));
    view.optionDataMemberViews.emplace_back(
            detail::extractSubOption(extractedOptions->batchUnrollCompileMethodSettings, *passOptions, strOptions));

    bool noneOptionExtracted =
            std::none_of(view.optionDataMemberViews.begin(), view.optionDataMemberViews.end(), [](const auto& opt) {
                return opt.has_value();
            });
    if (noneOptionExtracted) {
        return {};
    }
    view.guard = std::move(passOptions);
    view.optionDataPtr = std::move(extractedOptions);
    return view;
}

std::string BatchCompilerOptionsAdapterView::inject(const std::string& originalStrOptions) const {
    auto oldParams = BatchCompilerOptionsAdapterView::tryExtractFromString(originalStrOptions);
    std::string ret;
    if (!oldParams.has_value()) {
        ret = originalStrOptions + print();
    } else {
        // The idea is to cut out all suboptions  belonged to the `oldParams` from the initial string,
        // so that all other suboptions/option which are not related to BatchCompilerOptions will survice
        // After that we add `this` options print-result
        // It does make sense as the order of options used by through PipelineOptions is irrelevant,
        // so that instead of cutting and insetting updated values into related positions inside the original string
        // we could just cut old suboptions and add new one at the tail of the string
        auto greaterPos = std::not_fn(std::less<Occurence>());
        std::set<Occurence, decltype(greaterPos)> sortedDataMemberView(greaterPos);
        for (const auto& v : oldParams->optionDataMemberViews) {
            if (v.has_value()) {
                sortedDataMemberView.insert(v.value());
            }
        }
        // cut out relevant pipeline options starting from the end of the string
        ret = originalStrOptions;
        for (const auto& v : sortedDataMemberView) {
            ret = detail::cutOccurenceOutOfString(v, ret);
        }

        // append new values to the end of the string as an order is not important here
        // use space as a separator
        if (!ret.empty()) {
            ret += ' ';
        }
        ret += print();
    }
    return ret;
}

std::string BatchCompilerOptionsAdapterView::print() const {
    VPUX_THROW_UNLESS(optionDataPtr != nullptr, "BatchCompilerOptionsAdapterView::print() must have an object");
    VPUX_THROW_UNLESS(optionDataMemberViews.size() == 3,
                      "BatchCompilerOptionsAdapterView::print() view field count: {0} are not expected: {1}",
                      optionDataMemberViews.size(), 3);
    std::string ret;
    llvm::raw_string_ostream optionsPrinter(ret);
    guard->print(optionsPrinter);

    // remove leading '{' and trailing '}'
    if (ret.size() >= 2 && ret[0] == '{' and ret.back() == '}') {
        ret = ret.substr(1, ret.size() - 2);
    }
    return ret;
}

const BatchCompileOptionsAdapter& BatchCompilerOptionsAdapterView::get() const {
    VPUX_THROW_UNLESS(optionDataPtr != nullptr, "BatchCompilerOptionsAdapterView::get() must have an object");
    return *optionDataPtr;
}

DebatcherOptions::DebatcherOptions()
        : debatcherInliningMethod(*this, "debatching-inlining-method",
                                  llvm::cl::desc("Method for inlinging of debatching-function. Supported methods: "
                                                 "\"naive\", \"reordering\". Default is \"naive\""),
                                  llvm::cl::init("naive")),
          debatcherIntputCoeffPartitions(
                  *this, "debatcher-input-coefficients-partitions",
                  llvm::cl::desc(
                          "Determines which dimension and what proportion debatching of input tensors should be done. "
                          "Supported formats are: "
                          "\"input_name_0:[BatchDimension_0-DesiredBatchValueInDimension],input_name_1:["
                          "BatchDimension_1-DesiredBatchValueInDimension_1]\" "
                          "or for nameless inputs: "
                          "\"[BatchDimension_0-DesiredBatchValueInDimension],[BatchDimension_1-"
                          "DesiredBatchValueInDimension_1]\". For the last case the order is matter. "
                          "As default every 0-dimension considered as a BatchDimension and "
                          "DesiredBatchValueInDimension is become 1. The typical value of this options is "
                          "\"debatcher-input-coefficients-partitions=[0-1],[0-1],[0-1]\", which means that each tensor "
                          "of 2 inputs and 1 output will be debatched onto N=1, where N stays at 0 position starting "
                          "from the left side of a shape layout. In the layout NCHW the N has 0 position. Please be "
                          "noticed, that the default options are "
                          "not apt for more or less complicated networks!"),
                  llvm::cl::init("")),
          modelOpsNumberEnableThreshold(
                  *this, "model-ops-number-enable-threshold",
                  llvm::cl::desc(
                          "Determines a threshold as a minimum number of layers which a model has to consist of in "
                          "order "
                          "to activate \"debatch\" compilation method. "
                          "Although there is no hypothetical minimum or maximum of layers which amount prevents us "
                          "from "
                          "using \"debatch\" method (default=0), we have to take into account a real situation where "
                          "we deal with small models obtained from unit tests which affect a compilation time. "
                          "Eventually, the total resource consumption as well as an execution time inflates "
                          "drastically."
                          "Thus this parameters determines a low boundary at which \"debatch\" will be activated."
                          "If a model layer numbers doesn't met this condition, then the default compilation method "
                          "\"unroll\" will be employed."
                          "Please be notified: although no technical limitations exist from a pipeline perspective, "
                          "there will be non-zero threshold (typically 10)"),
                  llvm::cl::init(0)),
          maxBatchNumberDisableLimit(
                  *this, "max-batch-number-disable-limit",
                  llvm::cl::desc(
                          "Determines an upper boundary of a value in tensors batch dimension which being overstepped "
                          "prevents us from using \"debatch\" compilation method."
                          "As there is no theoretical minimum or maximum of supported batch values (default=MAX_UINT), "
                          "in a real situation we deal to various models obtained from unit tests, which batch "
                          "dimension may be enormous (e.g. 2000), and that affects a compilation time drastically. "
                          "Eventually, the total resource consumption as well as an execution time inflates above "
                          "normal."
                          "If max-batch size of any input in a model outruns this limit, then the default compilation "
                          "method \"unroll\" will be employed"
                          "Please be notified: although no technical limitations exist from a pipeline perspective, "
                          "there will be non-zero threshold (typically limited by TILES_NUMBER)"),
                  llvm::cl::init(std::numeric_limits<size_t>::max())) {
}

bool DebatcherOptions::isAvailable(const BatchCompileOptionsAdapter& options) {
    return options.batchCompileMethod == "debatch";
}

std::string DebatcherOptions::getDefaultOptions() {
    std::string ret;
    llvm::raw_string_ostream optionsPrinter(ret);
    DebatcherOptions{}.print(optionsPrinter);

    // remove leading '{' and trailing '}'
    if (ret.size() >= 2 && ret[0] == '{' and ret.back() == '}') {
        ret = ret.substr(1, ret.size() - 2);
    }

    return ret;
}

std::string DebatcherOptions::getDefaultDebatchInputCoeffPartitionsValue() {
    return DebatcherOptions{}.debatcherIntputCoeffPartitions;
}

std::unique_ptr<DebatcherOptions> DebatcherOptions::create(const BatchCompileOptionsAdapter& options) {
    if (!DebatcherOptions::isAvailable(options)) {
        return {};
    }
    VPUX_THROW_UNLESS(options.batchUnrollCompileMethodSettings.empty() ||
                              options.batchUnrollCompileMethodSettings == BatchUnrollOptions::getDefaultOptions(),
                      "DefaultHWOptionsBase is inconsistent: while \"{0}={1}\" was chosen,"
                      " non default value of the mutually exclusive option \"{2}\" has been detected: \"{3}\"",
                      options.batchCompileMethod.getArgStr(), options.batchCompileMethod,
                      options.batchUnrollCompileMethodSettings.getArgStr(), options.batchUnrollCompileMethodSettings);
    std::string settings = options.debatchCompileMethodSettings;

    return DebatcherOptions::createFromString(settings);
}

std::string DebatcherOptions::to_string() const {
    std::stringstream ss;
    ss << debatcherInliningMethod.getArgStr().data() << ": " << debatcherInliningMethod.getValue() << ", "
       << debatcherIntputCoeffPartitions.getArgStr().data() << ": " << debatcherIntputCoeffPartitions.getValue() << ", "
       << modelOpsNumberEnableThreshold.getArgStr().data() << ": " << modelOpsNumberEnableThreshold.getValue() << ", "
       << maxBatchNumberDisableLimit.getArgStr().data() << ": " << maxBatchNumberDisableLimit.getValue();
    return ss.str();
}

bool BatchUnrollOptions::isAvailable(const BatchCompileOptionsAdapter& options) {
    return options.batchCompileMethod == "unroll";
}

std::string BatchUnrollOptions::getDefaultOptions() {
    std::string ret;
    llvm::raw_string_ostream optionsPrinter(ret);
    BatchUnrollOptions{}.print(optionsPrinter);

    // remove leading '{' and trailing '}'
    if (ret.size() >= 2 && ret[0] == '{' and ret.back() == '}') {
        ret = ret.substr(1, ret.size() - 2);
    }

    return ret;
}

std::unique_ptr<BatchUnrollOptions> BatchUnrollOptions::create(const BatchCompileOptionsAdapter& options, Logger log) {
    if (!BatchUnrollOptions::isAvailable(options)) {
        return {};
    }
    log.info("Compilation proceeds with {0} options value 'unroll', settings: {1}",
             options.batchCompileMethod.getArgStr(), options.batchUnrollCompileMethodSettings);
    VPUX_THROW_UNLESS(options.debatchCompileMethodSettings.empty() ||
                              options.debatchCompileMethodSettings == DebatcherOptions::getDefaultOptions(),
                      "DefaultHWOptionsBase is inconsistent: while \"{0}={1}\" was chosen,"
                      " non default value of the mutually exclusive option \"{2}\" has been detected: \"{3}\"",
                      options.batchCompileMethod.getArgStr(), options.batchCompileMethod,
                      options.debatchCompileMethodSettings.getArgStr(), options.debatchCompileMethodSettings);
    std::string settings = options.batchUnrollCompileMethodSettings;

    return BatchUnrollOptions::createFromString(settings);
}
}  // namespace vpux
