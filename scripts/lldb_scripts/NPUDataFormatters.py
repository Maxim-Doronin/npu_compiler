#
# Copyright (C) 2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#
import lldb

# Define the constant for dynamic dimension representation
KDynamic = -(2**63)

# Mapping of _code values to their corresponding layout strings
dims_order_multimap = {
    0x1: ["C"],
    0x12: ["NC"],
    0x21: ["CN"],
    0x123: ["CHW"],
    0x231: ["HWC"],
    0x213: ["HCW"],
    0x321: ["WHC"],
    0x132: ["CWH"],
    0x1234: ["NCHW", "OIYX"],
    0x1342: ["NHWC", "OYXI"],
    0x1324: ["NHCW"],
    0x1432: ["NWHC"],
    0x1423: ["NWCH"],
    0x1243: ["NCWH"],
    0x4231: ["WCHN"],
    0x4312: ["WHNC"],
    0x3421: ["HWCN", "YXOI"],
    0x3214: ["HCNW"],
    0x3142: ["HNWC"],
    0x2413: ["CWNH"],
    0x2134: ["CNHW", "IOYX"],
    0x12345: ["NCDHW", "GNCHW", "GOIYX"],
    0x13452: ["NDHWC"],
    0x12453: ["GNHWC", "GOYXI"],
    0x14253: ["GHNWC"]
}


def DimsOrderSummaryProvider(valobj, internal_dict):
    """Provide a summary for vpux::DimsOrder by mapping _code to its layout string."""
    # Extract the _code field value
    code_value = valobj.GetChildMemberWithName('_code').GetValueAsUnsigned()

    # Map the code to its corresponding layout string
    layouts = dims_order_multimap.get(code_value, ["Unknown"])

    layout_str = "/".join(layouts)

    return f"DimsOrder(layout={layout_str})"


def ShapeSummaryProviderBase(valobj, container):
    """Provide a summary for vpux::Shape by extracting elements and determining dynamic/static status."""
    elements = []
    cont = valobj.GetChildMemberWithName(container)

    # Get the number of children in the container
    size = cont.GetNumChildren()
    if size > 4:
        return "{..}"  # Return a placeholder if there are more than 4 elements

    hasDynamicDims = False
    for i in range(size):
        element = cont.GetChildAtIndex(i)
        element_value = element.GetValue()
        # Check if the element value is dynamic
        if str(element_value) == str(KDynamic):
            elements.append("?")
            hasDynamicDims = True
        else:
            elements.append(str(element_value))

    # Join elements into a string representation
    elements_str = ",".join(elements)

    # Determine if the shape is dynamic or static
    dynamic_str = "dynamic" if hasDynamicDims else "static"
    return f"Shape(rank={size}, elements=[{elements_str}], {dynamic_str})"


def ShapeSummaryProvider(valobj, internal_dict):
    """Provide a summary for vpux::Shape using the base provider."""
    return ShapeSummaryProviderBase(valobj, '_cont')


def ShapeRefSummaryProvider(valobj, internal_dict):
    """Provide a summary for vpux::ShapeRef using the base provider."""
    return ShapeSummaryProviderBase(valobj, '_ref')


def find_value_recursively(valobj, value_name, deep=1):
    """Recursively search for a child named 'Value' and return its value."""
    # Limit recursion
    if deep > 2:
        return '{..}'
    # Check if the current object has a child named 'value_name'
    value_child = valobj.GetChildMemberWithName(value_name)
    if value_child is not None:
        if 'basic_string' in value_child.GetTypeName():
            return value_child.GetSummary()
        else:
            return value_child.GetValue()

    # If not, iterate over all children and search recursively
    for i in range(valobj.GetNumChildren()):
        child = valobj.GetChildAtIndex(i)
        value = find_value_recursively(child, value_name, deep + 1)
        if value is not None:
            return value

    # Return None if 'Value' is not found
    return None


def OptionSummaryProvider(valobj, internal_dict):
    """Provide a summary for various option types by comparing current and default values."""
    opt_storage = valobj.GetChildAtIndex(0).GetChildAtIndex(1)

    if 'basic_string' in opt_storage.GetTypeName():
        # Extract the current value as a string
        value = opt_storage.GetChildAtIndex(0).GetSummary()
    else:
        # Recursively find value in storage
        value = find_value_recursively(opt_storage, 'Value')

    # Extract the default value
    default_block = opt_storage.GetChildMemberWithName('Default')
    def_value = find_value_recursively(default_block, 'Value')

    # Format the summary based on whether the value is modified
    if value == def_value:
        res = f"DefValue: {def_value}"
    else:
        res = f"Value: {value} (modified), DefValue: {def_value}"

    return res


def __lldb_init_module(debugger, internal_dict):
    """Initialize the LLDB module and register type summaries for various types."""
    cat = debugger.CreateCategory("npu")
    cat.SetEnabled(True)

    # Register the summary provider for vpux::DimsOrder
    cat.AddTypeSummary(
        lldb.SBTypeNameSpecifier("vpux::DimsOrder", lldb.eFormatterMatchExact),
        lldb.SBTypeSummary.CreateWithFunctionName("NPUDataFormatters.DimsOrderSummaryProvider")
    )

    # Register summaries for Shape types
    cat.AddTypeSummary(
        lldb.SBTypeNameSpecifier("vpux::Shape", lldb.eFormatterMatchExact),
        lldb.SBTypeSummary.CreateWithFunctionName("NPUDataFormatters.ShapeSummaryProvider")
    )
    cat.AddTypeSummary(
        lldb.SBTypeNameSpecifier("vpux::details::ShapeTag<vpux::details::DimValuesBase<long> >",
                                 lldb.eFormatterMatchExact),
        lldb.SBTypeSummary.CreateWithFunctionName("NPUDataFormatters.ShapeSummaryProvider")
    )
    cat.AddTypeSummary(
        lldb.SBTypeNameSpecifier("vpux::ShapeRef", lldb.eFormatterMatchExact),
        lldb.SBTypeSummary.CreateWithFunctionName("NPUDataFormatters.ShapeRefSummaryProvider")
    )
    cat.AddTypeSummary(
        lldb.SBTypeNameSpecifier(
            "vpux::details::ShapeTag<vpux::details::DimValuesRefBase<long> >", lldb.eFormatterMatchExact),
        lldb.SBTypeSummary.CreateWithFunctionName("NPUDataFormatters.ShapeRefSummaryProvider")
    )

    # Register summaries for option types
    option_types = [
        "vpux::IntOption",
        "vpux::Int64Option",
        "vpux::StrOption",
        "vpux::BoolOption",
        "vpux::DoubleOption"
    ]
    for option_type in option_types:
        cat.AddTypeSummary(
            lldb.SBTypeNameSpecifier(option_type, lldb.eFormatterMatchExact),
            lldb.SBTypeSummary.CreateWithFunctionName("NPUDataFormatters.OptionSummaryProvider")
        )
