# VTune Guide
This guide will attempt to give an overview of the basics of setting up and
working with VTune.

**Note**: If you are using a developer VM and want to do remote profiling from Windows,
follow the next section to install VTune. Otherwise, you can skip it and read
the VTune documentation for a more standard installation.

## Installing VTune on a Developer VM
If you plan to use a VM for running VTune analysis, this section will
guide you through the steps for configuring remote sessions from a local
Windows box.

### Download VTune for Windows
- https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler-download.html

### Setup Linux Host for User-Mode Sampling
- https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2023-0/remote-linux-target-setup.html
- https://www.intel.com/content/www/us/en/docs/vtune-profiler/installation-guide/2023-0/package-managers.html

Hardware event-based sampling is not possible when running under a VM because we don't have access to hardware counters etc.
If you need this, you may want to look at setup on a local machine.

```
# Temporarily disregard normal proxy config
# You may need to run these commands as root since sudo doesn't preserve
# environment variables _or_ you can try `sudo -E`.
export no_proxy=

# Download and install Intel GPG key
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

# Add Intel repos to APT
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

# Fetch package lists from repos
sudo apt update

# Install VTune and related tools to enable user-mode sampling
sudo apt install intel-oneapi-vtune
sudo apt install linux-tools-common linux-tools-generic linux-tools-`uname -r`

# Check that everything is working (expect to see failures for hardware based features)
/opt/intel/oneapi/vtune/latest/bin64/vtune-self-checker.sh
```

## Standard Installation
Check VTune documentation [here](https://www.intel.com/content/www/us/en/docs/vtune-profiler/installation-guide/2023-0/overview.html)

## Profiling

### Prepare Application for Profiling
- https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2023-0/prepare-application.html

Application should be compiled with debug symbols. Using CMake, we can use
the build type: "RelWithDebInfo". You can create a user preset, inheriting
from vpuxDeveloper and simply changing the build type or pass the build
type manually.

```
cmake -DOpenVINODeveloperPackage_DIR=../../openvino/build --preset vpuxProfiler
```

##### Example CMake User Preset
This preset builds the project in release mode with debug symbols enabled (`-g`).

```json
{
  "name": "vpuxProfiler",
  "displayName": "vpuxProfiler",
  "description": "Build for use with a profiler",
  "binaryDir": "${sourceDir}/build-x86_64/Release",
  "inherits": [
    "vpuxDeveloper",
    "LinkerOptimization"
  ],
  "cacheVariables": {
    "CMAKE_CXX_FLAGS": "-g",
    "InferenceEngineDeveloperPackage_DIR": {
      "type": "FILEPATH",
      "value": "$env{OPENVINO_HOME}/build-x86_64/Release"
    },
    "CMAKE_BUILD_TYPE": {
      "type": "STRING",
      "value": "Release"
    },
    "ENABLE_DEVELOPER_BUILD": true,
    "ENABLE_CLANG_FORMAT": false,
    "ENABLE_VPUX_DOCS": false,
    "ENABLE_TESTS": true,
    "ENABLE_FUNCTIONAL_TESTS": true,
    "LIT_TESTS_USE_LINKS": true,
    "ENABLE_SPLIT_DWARF": true
  }
}
```

### Start Profiling   
- https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2023-0/analyze-performance.html

**Note**: Some of the following steps may change if you're running locally instead.

From your Windows instance of VTune, choose "Configure Analysis":

![image](images/vtune/configure-analysis.png)

Choose "Remote Linux (SSH)" and enter the details of your VM:

![image](images/vtune/remote-linux.png)

Specify application binary and command-line options:

![image](images/vtune/cmdline-options.png)

**Note**: Make sure to use absolute paths for all command-line options!

Most options can be left as default but take note of the following options:

![image](images/vtune/time-estimate.png)

Select an appropriate run time for the model you're compiling, you don't
have to be exact but lower number means higher sampling frequency. This can
influence the consistency of results for smaller models a fair amount.

![image](images/vtune/finalisation-mode.png)

Finalisation mode will affect how quickly metrics are generated but effectively
reduces the number of samples it pulls from. Generally you'll want to use
"Full" in most cases but again depends on size of model.

Lastly, make sure to use user-mode sampling and "Hotspots" analysis:

![image](images/vtune/hotspots.png)

Hotspots analysis highlights the most time consuming functions in your application.

### Interpreting Results
When you've finished collecting samples, you'll get a lot of data thrown at
you but the most useful tabs are "Caller/Callee" and "Top-down Tree".

![image](images/vtune/tabs.png)

using top-down tree, we can drill down into the call stack can find our passes:

![image](images/vtune/top-down-tree.png)

We want to filter CPU % to just those in the pipeline, choose "Filter In by Selection":

![image](images/vtune/filter-by-selection.png)

You can also filter using more options at the bottom of the window:

![image](images/vtune/top-down-tree-filter.png)

With DeepLabv3, we can see which passes are taking up the bulk of the runtime:

![image](images/vtune/deeplab-example.png)

### Comparing Results
- https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2023-0/comparing-results.html

You can compare the results of two runs by selecting them in the left-hand side panel:

![image](images/vtune/deeplab-comparison.png)

Select compare results:

![image](images/vtune/deeplab-comparison-2.png)

You can then take a look at "Caller/Callee" (or another tab like Top-Down Tree that we looked at before):

![image](images/vtune/caller-callee-tab.png)

Remembering to filter:

![image](images/vtune/caller-callee-filter.png)

We'll notice that there are a few columns to compare results, mostly we're interested in CPU%:

![image](images/vtune/caller-callee-cpu.png)

Now we can compare the performance characteristics between two profiles:

![image](images/vtune/compare-cpu.png)

## References
- https://www.youtube.com/watch?v=4jwhjsN_Ock
- https://www.youtube.com/watch?v=ghFn5IBzjrc
- Intel ITT can be used for instrumentation based profiling: https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2023-1/instrumentation-and-tracing-technology-apis.html
