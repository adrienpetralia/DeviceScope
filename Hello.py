# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
from utils import *

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="DeviceScope",
        page_icon=":zap:",
    )

    st.write("# DeviceScope")


    st.markdown(
        """
        Welcome to DeviceScope! :mag: :zap: This app provides an analytical tool to browse, detect and localize appliance patterns in electricity consumption time series.
        """
    )

    st.markdown(f"""
                1. **Explore** electrical consumption series at different sampling rates.
                2. **Detect** appliances in a period of time using the different trained classifiers and compare their performance.
                3. **Localize** appliance patterns using explainable classification approaches (CAM/AttMap).
                """)

    tab_playground, tab_benchmark, tab_about = st.tabs(
        ["Playground", "Benchmark", "About"]
    )
    
    with tab_playground:
        run_playground_frame()

    with tab_benchmark:
        run_benchmark_frame()

    with tab_about:
        run_about_frame()


if __name__ == "__main__":
    run()
