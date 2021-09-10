#ifndef CAN_INTERFACE_H
#define CAN_INTERFACE_H


#include <ros/ros.h>
#include <linux/can.h>
#include <string>

#include <<>AUTO_INSERT:PACKAGE_NAME<>/can_com.h>
#include <<>AUTO_INSERT:PACKAGE_NAME<>/can_encode_decode.h>

//#include <<>AUTO_INSERT:PACKAGE_NAME<>/can_message.h>
//autoInserted Messagetypes
//<>AUTO_INSERT:DO_PER_MESSAGE<>   #include <<>AUTO_INSERT:PACKAGE_NAME<>/<>AUTO_INSERT:MESSAGE_NAME<>.h>

//autoInserted Messagetypes for specifc rulesets
//<>AUTO_INSERT:RULESET_HEADERS<>   #include <<>AUTO_INSERT:PACKAGE_NAME<>/<>AUTO_INSERT:MESSAGE_NAME<>.h>


namespace <>AUTO_INSERT:PACKAGE_NAME<>
{

class can_com;

class CanInterface
{
public:
    CanInterface();
    CanInterface(const char* device);
    CanInterface(std::string& device);
    virtual ~CanInterface();

    bool initialize(const char* device);
    bool initialize(std::string& device);

private:
    void can_decode(struct can_frame& rec_frame);
    bool initialize_(std::string& device);
public:
private:
    bool initialized_;
    can_com* can_class;

    /* ###################################################### */
    /* S E N D I N G     T O     C A N */
    /* ###################################################### */
public:
    // Encode and Send Method Callbacks (from User to CAN)
    //<>AUTO_INSERT:DO_PER_MESSAGE<>   <W>      void send_<>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>(const <>AUTO_INSERT:PACKAGE_NAME<>::<>AUTO_INSERT:MESSAGE_NAME<> &msg);
  
    /* ###################################################### */
    /* R E C E I V I N G    F R O M     C A N */
    /* ###################################################### */
public:
    // Callback Handler Typedefs
    //<>AUTO_INSERT:DO_PER_MESSAGE<>   <R>      typedef std::function<void(const <>AUTO_INSERT:PACKAGE_NAME<>::<>AUTO_INSERT:MESSAGE_NAME<> &)> <>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>_callback_T;

    // Callback Registration Setter Functions (from CAN to User)
    //<>AUTO_INSERT:DO_PER_MESSAGE<>   <R>      void register_<>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>_callback(<>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>_callback_T);
private:
    // Callback Function Handlers used for registration above
    //<>AUTO_INSERT:DO_PER_MESSAGE<>   <R>      <>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>_callback_T receive_<>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>_callback_ = nullptr;
    /* ###################################################### */
};


} // end namespace

#endif // CAN_INTERFACE_H
