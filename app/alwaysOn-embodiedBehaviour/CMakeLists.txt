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


set(appname alwaysOn-embodiedBehaviour)

# Configure the correct path
set(resource_path ${CMAKE_INSTALL_PREFIX}/${ICUBCONTRIB_CONTEXTS_INSTALL_DIR}/${appname})

file(GLOB scripts ${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.xml)

if(scripts)
    yarp_install(FILES ${scripts} DESTINATION ${ICUBCONTRIB_APPLICATIONS_TEMPLATES_INSTALL_DIR})
    yarp_install(FILES ${scripts} DESTINATION ${ICUBCONTRIB_APPLICATIONS_INSTALL_DIR})
endif()
