# Copyright (c) 2026 Nima Abaeian
#
# Author: Nima Abaeian
# Organization: Istituto Italiano di Tecnologia
# Lab: Cognitive Architecture for Collaborative Technologies
# License: GNU GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


cmake_minimum_required(VERSION 3.12)

set(PYTHON_VERSION_arch "3.12" CACHE STRING "Python Version for alwaysOn modules")
set(PYTHON_VERSION ${PYTHON_VERSION_arch})
message("Use Python${PYTHON_VERSION}")

set(NAME_VENV ${PROJECT_NAME}_venv)
set(NAME_REQUIREMENTS "requirements")

set(PROGRAM_PERMISSIONS_DEFAULT
        OWNER_WRITE OWNER_READ OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE)

# Find Python and Virtualenv.
find_program(VIRTUALENV virtualenv)

if(NOT VIRTUALENV)
    message(FATAL_ERROR "Could not find `virtualenv`, please install it with `pip install virtualenv`")
endif()

set(ICUB_APPLICATIONS_PREFIX "$ENV{ICUB_ROOT}" CACHE PATH "Application path prefix")

# Find python command
execute_process(
    COMMAND bash -c "which python${PYTHON_VERSION}"
    OUTPUT_VARIABLE PYTHON_EXE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Print which python is being used
message(STATUS "BUILDING VIRTUAL ENV BASED ON PYTHON: ${PYTHON_EXE}")

# Generate the virtualenv and ensure it's up to date.
add_custom_command(
    OUTPUT ${NAME_VENV}
    COMMAND ${VIRTUALENV} --python=${PYTHON_EXE} ${NAME_VENV}
)

# Upgrade PIP
add_custom_command(
    OUTPUT upgrade_pip
    DEPENDS ${NAME_VENV}
    COMMAND ./${NAME_VENV}/bin/pip install --upgrade pip
)

# Install requirements from .txt file
add_custom_command(
    OUTPUT requirements
    DEPENDS upgrade_pip
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/${NAME_REQUIREMENTS}.txt ${CMAKE_CURRENT_BINARY_DIR}/${NAME_REQUIREMENTS}.txt
    COMMAND ./${NAME_VENV}/bin/pip install -r ${NAME_REQUIREMENTS}.txt
)

# Custom targets for CMake
# Create a virtual-env and install all requirements
add_custom_target(build-python ALL
    DEPENDS requirements
    SOURCES ${CMAKE_SOURCE_DIR}/${NAME_REQUIREMENTS}.txt
)

# set PATH_TO_VENV so that the python modules can find the virtualenv
set(PATH_TO_VENV ${CMAKE_CURRENT_BINARY_DIR}/${NAME_VENV})
message(STATUS "PATH_TO_VENV: ${PATH_TO_VENV}")

# Function to easily install your module Garello's Style
function(easy_python_module_install MODULE_NAME PYTHON_FILE)

    message("====================")
    message("Installing Module: ${MODULE_NAME}")
    message("Python File: ${PYTHON_FILE}")
    message("Python Version: ${PYTHON_VERSION}")
    message("Virtualenv: ${NAME_VENV}")
    message("Requirements File: ${NAME_REQUIREMENTS}.txt")

    # attach ".sh" to the end of the module name
    set(PATH_TO_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_FILE})
    set(SCRIPT_FILE_NAME ${MODULE_NAME})
    set(SCRIPT_CONTENT "#!/bin/bash\n\n# Function to terminate the child process when the bash father is terminated \nterminate_child_process() {\n    kill $PID\n}\n\n# Catch the quitting signal and send it to the child process\ntrap terminate_child_process EXIT\n# Activate virtual env\nsource ${PATH_TO_VENV}/bin/activate\n# Run python script\npython3 ${PATH_TO_SCRIPT} \"$@\"  &\nPID=$!\n\n# Wait for the python process to end\nwait $PID")

    # Generate the script file
    file(WRITE ${CMAKE_BINARY_DIR}/bin/${SCRIPT_FILE_NAME} ${SCRIPT_CONTENT})

    # Make the script file executable
    execute_process(COMMAND chmod +x ${CMAKE_BINARY_DIR}/bin/${SCRIPT_FILE_NAME})

    # Output instructions to the user
    message("Your ${SCRIPT_FILE_NAME} script has been created. You can now run it from terminal or yarpmanager.")

    message(STATUS "CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})
    message(STATUS "Installing ${MODULE_NAME} in ${ICUBcontrib_DIR}/bin/${SCRIPT_FILE_NAME}")

    install(FILES ${CMAKE_BINARY_DIR}/bin/${SCRIPT_FILE_NAME} PERMISSIONS ${PROGRAM_PERMISSIONS_DEFAULT} DESTINATION bin)

endfunction()

# ==========================================================================
easy_python_module_install(alwayson_vision vision.py)
easy_python_module_install(alwayson_executiveControl executiveControl.py)
easy_python_module_install(alwayson_salienceNetwork salienceNetwork.py)
easy_python_module_install(alwayson_chatBot chatBot.py)
