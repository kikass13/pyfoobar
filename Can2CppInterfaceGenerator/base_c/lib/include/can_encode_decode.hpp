#ifndef CAN_ENCODE_DECODE_INL_H_
#define CAN_ENCODE_DECODE_INL_H_

namespace <>AUTO_INSERT:PACKAGE_NAME<>
{

#include <stdint.h> //uint typedefinitions, non-rtw!

#define MASK64(nbits) ((0xffffffffffffffff)>> (64-nbits))

float toPhysicalValue(uint64_t target, float factor, float offset, bool is_signed);
uint64_t fromPhysicalValue(float physical_value, float factor, float offset);
void storeSignal(uint8_t* frame, uint64_t value, const uint8_t startbit, const uint8_t length, bool is_big_endian, bool is_signed);
uint64_t extractSignal(const uint8_t* frame, const uint8_t startbit, const uint8_t length, bool is_big_endian, bool is_signed);
float decode(const uint8_t* frame, const uint16_t startbit, const uint16_t length, bool is_big_endian, bool is_signed, float factor, float offset);
void encode(uint8_t* frame, const float value, const uint16_t startbit, const uint16_t length, bool is_big_endian, bool is_signed, float factor, float offset);

#endif



} // end namespace