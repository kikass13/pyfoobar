# ntrip_client

## Overview

Ntrip client based on Oxford Ntrip Client
https://bitbucket.org/DataspeedInc/oxford_gps_eth/src/ntrip_forward/src/oxford_gps_eth/ntrip_forwarding.py

## SAPOS Accounts

| Account      | Vehicle          |
|---|---|
|         |         |
|         |         |
|         |         |
|         |         |
|         |         |
|         |         |

## Installation

### Building from Source

#### Dependencies

- [Robot Operating System (ROS)](http://wiki.ros.org) (middleware for robotics),
- Python2.7
- User shall be a member of dialout group (otherwise the user has no permission to send data to the ttyUSB port)


#### Building

To build from source, clone the latest version from this repository into your catkin workspace and compile the package using

	cd catkin_ws/src
	git clone ...
	cd ../
	catkin build

## Usage

Run the main node with

	rosrun ntrip_client ntrip_client.py


