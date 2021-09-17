#pragma once

#include <stdint.h>

struct CanFrame{
    uint16_t id;
    uint8_t* data;
    uint8_t dlc;
};

namespace <>AUTO_INSERT:PACKAGE_NAME<>
{
    //<>AUTO_INSERT:MESSAGE_DECODE_ALL<>
    struct  <>AUTO_INSERT:MESSAGE_NAME<>
    {
        //do this for all signals
            //<>AUTO_INSERT:SIGNAL_DECODE<>
            /// startbit, length, isBigEndian, isSigned, factor, offset
            /// <>AUTO_INSERT:SIGNAL_STARTBIT<>, <>AUTO_INSERT:SIGNAL_LENGTH<>, <>AUTO_INSERT:SIGNAL_ISBIGENDIAN<>, <>AUTO_INSERT:SIGNAL_ISSIGNED<>, <>AUTO_INSERT:SIGNAL_FACTOR<>, <>AUTO_INSERT:SIGNAL_OFFSET<> );
            <>AUTO_INSERT:SIGNAL_TYPE<> <>AUTO_INSERT:SIGNAL_NAME<>;
            //<>AUTO_INSERT:SIGNAL_DECODE<>
    };
    //<>AUTO_INSERT:MESSAGE_DECODE_ALL<>

}// end namespace