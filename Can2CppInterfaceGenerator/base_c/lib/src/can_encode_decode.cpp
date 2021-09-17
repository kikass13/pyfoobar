#include <<>AUTO_INSERT:PACKAGE_NAME<>/can_encode_decode.hpp>
#include <stdint.h> //uint typedefinitions, non-rtw!


namespace <>AUTO_INSERT:PACKAGE_NAME<>
{

float toPhysicalValue(uint64_t target, float factor, float offset, bool is_signed)
{
    if (is_signed) {
        //ROS_INFO("Ende: %f",( (int32_t) target ) * factor + offset);
	return (( (int32_t) target ) * factor + offset);
    } else {
        return target * factor + offset;
    }
}

uint64_t fromPhysicalValue(float physical_value, float factor, float offset)
{
    return (physical_value - offset) / factor;
}

void storeSignal(uint8_t* frame, uint64_t value, const uint8_t startbit, const uint8_t length, bool is_big_endian, bool is_signed)
{
    uint8_t start_byte = startbit / 8;
    uint8_t startbit_in_byte = startbit % 8;
    uint8_t end_byte = 0;
    int8_t count = 0;
    uint8_t current_target_length = (8-startbit_in_byte);

    // Mask the value
    value &= MASK64(length);

    // Write bits of startbyte
    frame[start_byte] |= value << startbit_in_byte;

    // Write residual bytes
    if(is_big_endian) // Motorola (big endian)
    {
        end_byte = (start_byte * 8 + 8 - startbit_in_byte - length) / 8;

        for(count = start_byte-1; count >= end_byte; count --)
        {
            frame[count] |= value >> current_target_length;
            current_target_length += 8;
        }
    }
    else // Intel (little endian)
    {
        end_byte = (startbit + length - 1) / 8;

        for(count = start_byte+1; count <= end_byte; count ++)
        {
            frame[count] |= value >> current_target_length;
            current_target_length += 8;
        }
    }
}

uint64_t extractSignal(const uint8_t* frame, const uint8_t startbit, const uint8_t length, bool is_big_endian, bool is_signed)
{
    // Init with all zero
    uint8_t start_byte = 0;
    uint8_t startbit_in_byte = 0;
    uint8_t current_target_length = 0;
    uint8_t end_byte = 0;
    int8_t count = 0;   
    uint64_t target = 0;
    
    // Write bytes 
    if(is_big_endian) // Motorola (big endian)
    {
        uint8_t startbit_in_byte = (startbit+length) % 8;
        uint8_t current_target_length = (8-startbit_in_byte);
        start_byte = (startbit + length - 1)/8;
	end_byte =  startbit / 8;
        target = frame[start_byte] >> startbit_in_byte;
        for(count = start_byte-1; count >= end_byte; count --)
        {
            target |= frame[count] << current_target_length;
            current_target_length += 8;
        }    	
    }
    else // Intel (little endian)
    {
        uint8_t startbit_in_byte = startbit % 8;
        uint8_t current_target_length = (8-startbit_in_byte);
        start_byte = startbit / 8 ;
        end_byte = (startbit + length - 1) / 8;
        target = frame[start_byte] >> startbit_in_byte;
        for(count = start_byte+1; count <= end_byte; count ++)
        {
            target |= frame[count] << current_target_length;
            current_target_length += 8;
        }
    }

    // Mask value
    target &= MASK64(length);

    // perform sign extension
    if (is_signed)
    {
        int64_t msb_sign_mask = 1 << (length - 1);
	//ROS_INFO("Vorher: %li",target);
        target = ( (int32_t) target ^ msb_sign_mask) - msb_sign_mask;
        //ROS_INFO("Nacher: %li",target);
    }

    return target;
}

float decode(const uint8_t* frame, const uint16_t startbit, const uint16_t length, bool is_big_endian, bool is_signed, float factor, float offset)
{
    return toPhysicalValue(extractSignal(frame, startbit, length, is_big_endian, is_signed), factor, offset, is_signed);
}

void encode(uint8_t* frame, const float value, const uint16_t startbit, const uint16_t length, bool is_big_endian, bool is_signed, float factor, float offset)
{
    storeSignal(frame, fromPhysicalValue(value, factor, offset), startbit, length, is_big_endian, is_signed);
}


} // end namespace