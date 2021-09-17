
#include <<>AUTO_INSERT:PACKAGE_NAME<>/can_interface.hpp>


//<>AUTO_INSERT:MESSAGE_RULESET_VARIABLES<> 

namespace <>AUTO_INSERT:PACKAGE_NAME<>
{

CanInterface::CanInterface(){
}

CanInterface::~CanInterface(){
}


// Callback Registration Setter Functions (from CAN to User)
//<>AUTO_INSERT:DO_PER_MESSAGE<>   <R>      void CanInterface::register_<>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>_callback(<>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>_callback_T cb){this->receive_<>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>_callback_ = cb;}



//<>AUTO_INSERT:MESSAGE_ENCODE<>
void CanInterface::send_<>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>(const <>AUTO_INSERT:PACKAGE_NAME<>::<>AUTO_INSERT:MESSAGE_NAME<> &msg)
{
    CanFrame canMsg;

    canMsg.id= <>AUTO_INSERT:MESSAGE_ID<>;
    canMsg.dlc = <>AUTO_INSERT:MESSAGE_DLC<>;
    memset(&canMsg.data[0], 0, sizeof(canMsg.data));
    unsigned int base_counter, byte_counter = 0;
    long conversion;
    uint8_t* ptr;
    //<>AUTO_INSERT:SIGNAL_ENCODE<>
    conversion = (uint64_t) msg.<>AUTO_INSERT:SIGNAL_NAME<>;
    ptr = (uint8_t*) &conversion;
    base_counter = 0;
    for (unsigned int i = <>AUTO_INSERT:SIGNAL_STARTBIT<>; i <   <>AUTO_INSERT:SIGNAL_STARTBIT<> + <>AUTO_INSERT:SIGNAL_LENGTH<>; ++i){
        byte_counter = (unsigned int) i/8;
        bool tmp_bit = (*ptr >> base_counter) & 0x01;
        canMsg.data[byte_counter] = canMsg.data[byte_counter] | (tmp_bit << i);
        base_counter++;
    }
    //<>AUTO_INSERT:SIGNAL_ENCODE<>
    if(this->send_callback_ != nullptr)
        this->send_callback_(canMsg);
}
//<>AUTO_INSERT:MESSAGE_ENCODE<>

void CanInterface::read(const CanFrame& msg)
{
    /*CREATE THIS BLOB*/
    switch(msg.id)
    {
    //<>AUTO_INSERT:MESSAGE_DECODE<>
    case  <>AUTO_INSERT:MESSAGE_ID<> :
    {
        <>AUTO_INSERT:PACKAGE_NAME<>::<>AUTO_INSERT:MESSAGE_NAME<> var_<>AUTO_INSERT:MESSAGE_NAME<>;
        //insert signals here
        //add variables into message object
        //do this for all signals
            //<>AUTO_INSERT:SIGNAL_DECODE<>
            <>AUTO_INSERT:SIGNAL_TYPE<> <>AUTO_INSERT:SIGNAL_NAME<> = decode (msg.data, <>AUTO_INSERT:SIGNAL_STARTBIT<>, <>AUTO_INSERT:SIGNAL_LENGTH<>, <>AUTO_INSERT:SIGNAL_ISBIGENDIAN<>, <>AUTO_INSERT:SIGNAL_ISSIGNED<>, <>AUTO_INSERT:SIGNAL_FACTOR<>, <>AUTO_INSERT:SIGNAL_OFFSET<> );
            var_<>AUTO_INSERT:SIGNAL_PARENT_MESSAGE<>.<>AUTO_INSERT:SIGNAL_NAME<> = <>AUTO_INSERT:SIGNAL_NAME<>;
            //<>AUTO_INSERT:SIGNAL_DECODE<>
        
        if(this->receive_<>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>_callback_ != nullptr)
            this->receive_<>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>_callback_(var_<>AUTO_INSERT:MESSAGE_NAME<>);

        //<>AUTO_INSERT:MESSAGE_RULESET_MESSAGE_CONVERSION<>
/*
        //<>AUTO_INSERT:MESSAGE_RULESET_MESSAGE_PUBLISH<>
*/
        break;
    }
    //<>AUTO_INSERT:MESSAGE_DECODE<>

    default:
        break;
    }
}



} // end namespace