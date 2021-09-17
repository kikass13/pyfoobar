# General

This python2.7 tool will generate a ROS-Node from a CAN database file (.dbc) containing all relevant message definitions, subscribers (CAN WRITE) and publishers (CAN READ).

The parser can't parse all dbc commands (at the moment) ... these types for example have to be removed from the .dbc file before parsing it.

``
VAL_
BA_
BA_DEF
CM_
``
#

# Setup

```
sudo apt-get pip install python-pip
pip install --upgrade pip
pip install cantools 
```

# How to use

## Configure Package 
First run this script with the command `configure` and provide package_name and the paths to .dbc files.
If given multiple .dbc files, the messages of those files will be generate separately, whereby each message of a .dbc file will be put into a seperate ros namespace.

```
python dbc.py configure myPackage examples/testDbcFile1.dbc examples/testDbcFile2.dbc
```

Multiple configuration files will be generated (.cfg, .db, .specs) which define the code generation process. The .cfg file is the main configuration file and will specify (link) the other files.

After successful configuration, we can alter the signals and messages provided by the .dbc files inside the .specs configuration. We could, for example, change the directions of specific messages from read("r") to write("w").

You can also specify a ros message conversion for a specific signal by defining a ruleset (in the .rules file) like so

```
[
    {
        "id": "someRandomeConversionStuff1",
        "ns": "geometry_msgs",
        "type": "PoseStamped",
        "publish_trigger": "EncoderCounter",
        "rules": 
        [
            {
                "in": "SwitchAll",
                "set": "header.stamp = stamp"
            },
            {
                "in": "SwitchAll",
                "set": "pose.position.x = Checksum"
            },
            {
                "in": "EncoderCounter",
                "set": "pose.position.y = Encoder3"
            },
            {
                "in": "EncoderCounter",
                "set": "pose.position.z = Encoder4"
            }
        ]
    }
]
```

which will be put into code similiar to this:

```
geometry_msgs::PoseStamped someRandomeConversionStuff1_var ;

// ...

void can::publish_data(struct can_frame& rec_frame)
{
   // ...
   
   // Conversion aggregations are specified here ...
    someRandomeConversionStuff1_var.pose.position.y = Encoder3 ;
    someRandomeConversionStuff1_var.pose.position.z = Encoder4 ;

   // Conversion publishers are specified here ...
    pub_testDbcFile1_someRandomeConversionStuff1.publish(someRandomeConversionStuff1_var) ;
```

## Generate Package Node

To generate the ros node and all other relevant outputs, we have to use the `generate` command and provide a valid .cfg configuration file.

```
python dbc.py generate node outputs/myPackage.cfg
```

The output node will be generated in the "outputs" directory. You can put this node inside a catkin workspace and build/run it. Please remember, that you have to provide a parameter containing the linux device name of your can interface (`/myPackage/device`).

At the moment you have to manually generate a .launch file which starts this node (see launchfile template in the "base_c/launch" directory)

## Generate Package Library

To generate the ros interface library and all other relevant outputs, we have to use the `generate` command and provide a valid .cfg configuration file.

```
python dbc.py generate lib outputs/myPackage.cfg
```

The output library package will be generated in the "outputs" directory. You can put this node inside a catkin workspace and build/include it. Please remember, that you have to provide a parameter containing the linux device name of your can interface.

A generated howto inside can be found in the generated README.md file inside of the package.


# How  does it work

The tool will use the "base_c" directory, which contains a standard ROS catkin package to create a new ROS package inside the "outputs" directory. It does that by searching for keywords like `<>AUTO_INSERT:PACKAGE_NAME<>` and replacing them with proper information, either from user input/configuration or from provided .dbc files. This is done inside the package definition (CMakeLists.txt, package.xml) and inside the code files (/include, /src). 

The default code is also used to generate callbacks and buffers (subscriber) for CAN writing and also data movers and senders (publisher) for CAN reading. The provided .dbc information is used to generate appropriate bitshifters, endians, sizes and floating conversions.
