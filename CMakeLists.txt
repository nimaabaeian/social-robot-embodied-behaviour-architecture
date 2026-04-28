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
project(alwaysOnEmbodiedBehaviour)

find_package(ICUBcontrib REQUIRED)
find_package(YARP REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${ICUBCONTRIB_MODULE_PATH})

include(ICUBcontribHelpers)
include(ICUBcontribOptions)
icubcontrib_set_default_prefix()

add_subdirectory(modules)
add_subdirectory(app)

icubcontrib_add_uninstall_target()
