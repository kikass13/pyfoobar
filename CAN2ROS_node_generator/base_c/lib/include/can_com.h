#ifndef MYCAN_H
#define MYCAN_H

#include <boost/asio.hpp>
#include <boost/bind.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/thread.hpp>

#include <linux/can.h>
//#include <linux/can/raw.h>



namespace <>AUTO_INSERT:PACKAGE_NAME<>
{


class can_com
{
    typedef std::function<void(struct can_frame&)> CanDecodeHandler_T;

private:
    struct can_frame send_frame;
    struct can_frame rec_frame;

public:    

    boost::asio::io_service my_io_service;

    can_com(const char*, CanDecodeHandler_T decode_callback);
    virtual ~can_com(void);

    void send(struct can_frame& send_frame);

    void data_sent(const boost::system::error_code& error, std::size_t bytes_transferred);
    void data_received(struct can_frame& rec_frame,boost::asio::posix::basic_stream_descriptor<>& stream);

    void stopComm(void);
    void startComm(void);

    bool isInitialized();
    
private:
    bool initialized_;

    boost::asio::posix::basic_stream_descriptor<> stream;


    CanDecodeHandler_T decode_can_handler_;

};


} // end namespace


#endif // CAN_H