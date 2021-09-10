
#include <<>AUTO_INSERT:PACKAGE_NAME<>/can_interface.h>


//<>AUTO_INSERT:MESSAGE_RULESET_VARIABLES<> 

namespace <>AUTO_INSERT:PACKAGE_NAME<>
{


CanInterface::CanInterface() : initialized_(false){}
CanInterface::CanInterface(const char* device) : initialized_(false){
    this->initialize(device);
}
CanInterface::CanInterface(std::string& device) : initialized_(false){
    this->initialize(device);
}

CanInterface::~CanInterface(){
    if(this->initialized_) 
        delete this->can_class;
}

bool CanInterface::initialize(const char* device){
    std::string s(device);
    return this->initialize_(s);
}
bool CanInterface::initialize(std::string& device){
    return this->initialize_(device);
}

bool CanInterface::initialize_(std::string& device){
    if(!this->initialized_){
        // instantiate our can communication object and register a decoding function for it
        can_class = new can_com(device.c_str(), std::bind(&CanInterface::can_decode, this, std::placeholders::_1));
        if(can_class->isInitialized())
            this->initialized_ = true;
    }
    return this->initialized_;
}



// Callback Registration Setter Functions (from CAN to User)
//<>AUTO_INSERT:DO_PER_MESSAGE<>   <R>      void CanInterface::register_<>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>_callback(<>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>_callback_T cb){this->receive_<>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>_callback_ = cb;}



//<>AUTO_INSERT:MESSAGE_ENCODE<>
void CanInterface::send_<>AUTO_INSERT:PACKAGE_NS<>_<>AUTO_INSERT:MESSAGE_NAME<>(const <>AUTO_INSERT:PACKAGE_NAME<>::<>AUTO_INSERT:MESSAGE_NAME<> &msg)
{
    struct can_frame can_message;

    can_message.can_id = <>AUTO_INSERT:MESSAGE_ID<>;
    can_message.can_dlc = <>AUTO_INSERT:MESSAGE_DLC<>;
    std::memset(&can_message.data[0], 0, sizeof(can_message.data));
    unsigned int base_counter, byte_counter = 0;
    long conversion;
    unsigned char* ptr;
    //<>AUTO_INSERT:SIGNAL_ENCODE<>
    conversion = (long) msg.<>AUTO_INSERT:SIGNAL_NAME<>;
    ptr = (unsigned char*) &conversion;
    base_counter = 0;
    for (unsigned int i = <>AUTO_INSERT:SIGNAL_STARTBIT<>; i <   <>AUTO_INSERT:SIGNAL_STARTBIT<> + <>AUTO_INSERT:SIGNAL_LENGTH<>; ++i){
        byte_counter = (unsigned int) i/8;
        bool tmp_bit = (*ptr >> base_counter) & 0x01;
        can_message.data[byte_counter] = can_message.data[byte_counter] | (tmp_bit << i);
        base_counter ++;
    }
    //<>AUTO_INSERT:SIGNAL_ENCODE<>
    can_class->send(can_message);
}
//<>AUTO_INSERT:MESSAGE_ENCODE<>

void CanInterface::can_decode(struct can_frame& rec_frame)
{
    /*CREATE THIS BLOB*/

    switch(rec_frame.can_id)
    {
    //<>AUTO_INSERT:MESSAGE_DECODE<>
    case  <>AUTO_INSERT:MESSAGE_ID<> :
    {
        <>AUTO_INSERT:PACKAGE_NAME<>::<>AUTO_INSERT:MESSAGE_NAME<> var_<>AUTO_INSERT:MESSAGE_NAME<>;
        //insert signals here
        //add variables into message object
        //do this for all signals
            //<>AUTO_INSERT:SIGNAL_DECODE<>
            <>AUTO_INSERT:SIGNAL_TYPE<> <>AUTO_INSERT:SIGNAL_NAME<> = decode (rec_frame.data, <>AUTO_INSERT:SIGNAL_STARTBIT<>, <>AUTO_INSERT:SIGNAL_LENGTH<>, <>AUTO_INSERT:SIGNAL_ISBIGENDIAN<>, <>AUTO_INSERT:SIGNAL_ISSIGNED<>, <>AUTO_INSERT:SIGNAL_FACTOR<>, <>AUTO_INSERT:SIGNAL_OFFSET<> );
            var_<>AUTO_INSERT:SIGNAL_PARENT_MESSAGE<>.<>AUTO_INSERT:SIGNAL_NAME<> = <>AUTO_INSERT:SIGNAL_NAME<>;
            //<>AUTO_INSERT:SIGNAL_DECODE<>
        
        ros::Time stamp = ros::Time::now();
        var_<>AUTO_INSERT:MESSAGE_NAME<>.header.stamp = stamp;
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