#ifndef CAN_INTERFACE_H
#define CAN_INTERFACE_H

#include <stdint.h>
#include <functional>
#include <string.h>

#include <<>AUTO_INSERT:PACKAGE_NAME<>/can_encode_decode.hpp>
#include <<>AUTO_INSERT:PACKAGE_NAME<>/can_types.hpp>

//autoInserted Messagetypes for specifc rulesets
//<>AUTO_INSERT:RULESET_HEADERS<>   #include <<>AUTO_INSERT:PACKAGE_NAME<>/<>AUTO_INSERT:MESSAGE_NAME<>.h>


namespace <>AUTO_INSERT:PACKAGE_NAME<>
{



class CanInterface
{
public:
    CanInterface();
    virtual ~CanInterface();

private:
    void read(const CanFrame& msg);
public:
private:

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
    typedef std::function<void(const CanFrame&)> ExternalCanWriteCallback; 
    // Callback Handler Typedefs
    //<>AUTO_INSERT:DO_PER_MESSAGE<>   <R>      typedef std::function<void(const <>AUTO_INSERT:PACKAGE_NAME<>::<>AUTO_INSERT:MESSAGE_NAME<> &)> <>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>_callback_T;

    // Callback Registration Setter Functions (from CAN to User)
    //<>AUTO_INSERT:DO_PER_MESSAGE<>   <R>      void register_<>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>_callback(<>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>_callback_T);
private:
    // Callback Function Handlers used for registration above
    //<>AUTO_INSERT:DO_PER_MESSAGE<>   <R>      <>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>_callback_T receive_<>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>_callback_ = nullptr;
    /* ###################################################### */

    void registerSendCallback(ExternalCanWriteCallback cb) { send_callback_ = cb;}
    ExternalCanWriteCallback send_callback_;
};


} // end namespace

#endif // CAN_INTERFACE_H
