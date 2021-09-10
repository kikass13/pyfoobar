#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <net/if.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>

#include <linux/can.h>
#include <linux/can/raw.h>

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>


#include <ros/ros.h>

#include <<>AUTO_INSERT:PACKAGE_NAME<>/can_com.h>



namespace <>AUTO_INSERT:PACKAGE_NAME<>
{

can_com::can_com(const char* device, CanDecodeHandler_T decode_callback) : stream(my_io_service), initialized_(false)
{
    int iSock;
    struct sockaddr_can addr;

    struct ifreq ifr;

    iSock = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (iSock < 0)
    {
        ROS_ERROR("Error in socket bind");
    }

    addr.can_family=AF_CAN;


    addr.can_ifindex = if_nametoindex((const char *)device);
    if (addr.can_ifindex==0)
    {
        ROS_ERROR("device not found!");
        return;
    }
    bind(iSock,(struct sockaddr *)&addr, sizeof(addr));


    stream.assign(iSock);

    // enable input filter for each receiving message
    struct can_filter rfilter[<>AUTO_INSERT:MESSAGE_COUNT<><R>];

    //<>AUTO_INSERT:DO_PER_MESSAGE<>    <R>  rfilter[<>AUTO_INSERT:INDEX<>].can_id = <>AUTO_INSERT:MESSAGE_ID<>;
    //<>AUTO_INSERT:DO_PER_MESSAGE<>    <R>  rfilter[<>AUTO_INSERT:INDEX<>].can_mask = CAN_SFF_MASK;
    
    setsockopt(iSock, SOL_CAN_RAW, CAN_RAW_FILTER, &rfilter, sizeof(rfilter));

    // set our given decode callback
    this->decode_can_handler_ = decode_callback;

    this->initialized_ = true;
    startComm();
}

can_com::~can_com(){
    ROS_INFO("Destructor can");
    this->stopComm();
}

bool  can_com::isInitialized(){
    return this->initialized_;
}


void can_com::send(struct can_frame& send_frame)
{
    stream.async_write_some(boost::asio::buffer(&send_frame, sizeof(send_frame)),
                            boost::bind(&can_com::data_sent,this,
                                boost::asio::placeholders::error,
                                boost::asio::placeholders::bytes_transferred));
}


void can_com::data_sent(const boost::system::error_code& error, // Result of operation.
                        std::size_t bytes_transferred)           // Number of bytes written.)
{
    std::cout << "CAN: data sent, error code: " << error << ", bytes written: " << bytes_transferred << std::endl;
}

void can_com::data_received(struct can_frame& rec_frame,boost::asio::posix::basic_stream_descriptor<>& stream)
{
     /*std::cout << std::hex << rec_frame.can_id << "  ";
     for(int i=0;i<rec_frame.can_dlc;i++)
     {
         std::cout << std::hex << int(rec_frame.data[i]) << " ";
     }
     std::cout << std::dec << std::endl;
     */

     /*
      * Controller Area Network Identifier structure
      *
      * bit 0-28	: CAN identifier (11/29 bit)
      * bit 29	: error message frame flag (0 = data frame, 1 = error message)
      * bit 30	: remote transmission request flag (1 = rtr frame)
      * bit 31	: frame format flag (0 = standard 11 bit, 1 = extended 29 bit)
      */
     if (rec_frame.can_id & CAN_EFF_FLAG)
         ROS_WARN("CAN extended frame format!");
     else if (rec_frame.can_id & CAN_RTR_FLAG){
         ROS_WARN("CAN remote transmission request!");
         ROS_INFO("Did you set bitrate and bring can up?\n"
                  "Try:\n"
                  "   sudo ip link set can0 up type can bitrate 500000\n"
                  "   sudo ifconfig can0 up");
     }
     else if (rec_frame.can_id & CAN_ERR_FLAG)
        ROS_WARN("CAN ERROR!");
     else{
        this->decode_can_handler_(rec_frame);
     }

     // start new async read and wait for callback
     stream.async_read_some(
                 boost::asio::buffer(&rec_frame, sizeof(rec_frame)),
                 boost::bind(&can_com::data_received,this,
                             boost::ref(rec_frame),boost::ref(stream)));
}


void can_com::startComm(void)
{
    //stream.async_write_some(boost::asio::buffer(&frame, sizeof(frame)),boost::bind(data_send));
    stream.async_read_some(
                boost::asio::buffer(&rec_frame, sizeof(rec_frame)),
                boost::bind(&can_com::data_received,this,
                            boost::ref(rec_frame),boost::ref(stream)));


    // start thread and bind it to "io_service.run" (prevents blocking call "io_service.run")
    boost::thread bt(boost::bind(&boost::asio::io_service::run, &my_io_service));

    ROS_INFO("can_interface: Started listening...");
}

void can_com::stopComm(void)
{
    ROS_INFO("can_interface: Stop Communication...");
    //std::cout << "stop...\n";

    my_io_service.stop();

    stream.cancel();
    stream.close();

}




} // end namespace