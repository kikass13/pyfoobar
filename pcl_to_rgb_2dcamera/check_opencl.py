import pyopencl as cl
import numpy as np

platforms = cl.get_platforms()
print("Available OpenCL Platforms:")
for platform in platforms:
    print(f"  - {platform.name}")
print(platform.get_devices())

# Create OpenCL context and command queue
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Print device information
print("Device Name:", device.name)
print("Device Vendor:", device.vendor)
print("Device Version:", device.version)
print("Driver Version:", device.driver_version)
print("Device Profile:", device.profile)
print("Device Type:", cl.device_type.to_string(device.type))
print("Max Compute Units:", device.max_compute_units)
print("Max Work Group Size:", device.max_work_group_size)
print("Max Work Item Sizes:", device.max_work_item_sizes)
print("Global Memory Size:", device.global_mem_size)
print("Local Memory Size:", device.local_mem_size)
print("Max Clock Frequency:", device.max_clock_frequency)
print("Extensions:", device.extensions)

print("=========================================")
# Replace this with your actual point structure and data
# For example, assuming each point is a 3D coordinate represented by float32 values
point_dtype = np.dtype(np.float32)
point_size = 3 * point_dtype.itemsize  # Assuming each point is a 3D coordinate

global_dimensions = 1000000 #n points

a,b,c = device.max_work_item_sizes
n = a*b 
print("MAXIMUM POINTS PER QUERY POSSIBLE = %s" % n)

queriesNeeded = np.ceil(global_dimensions / n) + 1
print("QUERYS NEEDED (+safety/overhead margin) = %s" % queriesNeeded)

print("==============================================================")
# Check if the command queue supports out-of-order execution
out_of_order_exec = queue.properties & cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE != 0
# Print the result
if out_of_order_exec:
    print("Command queue supports out-of-order execution.")
else:
    print("Command queue does not support out-of-order execution.")
