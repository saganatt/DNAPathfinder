#ifndef __COLOR_GENERATOR_H__
#define __COLOR_GENERATOR_H__

#include <string>
#include <iostream>
#include <iomanip>
#include <helper_math.h>

class ColorGenerator {
    private:
        uint32_t numberOfColors;                // How many different colors will be generated
        // Colors generated in HSL, S = L = 0.6, H picked uniformly from [0, 1)
        // For Chimera representation they need to be converted to RGB
        // Helper variables:
        const float q = 0.84f;                  // q = L + S - L * S
        const float p = 0.36f;                  // p = 2 * L - q
        const uint32_t maxValue = 65536;        // max for R, G, B in Chimera

        // Get RGB color at index
        int3 getNextRGBColor(uint32_t index) {
            float h = (1.0f * index) / numberOfColors;
            // RGB in [0, 65535)
            uint32_t r = (uint32_t)roundf(hue2RGB(h + 1.0f / 3.0f) * maxValue);
            uint32_t g = (uint32_t)roundf(hue2RGB(h) * maxValue);
            uint32_t b = (uint32_t)roundf(hue2RGB(h - 1.0f / 3.0f) * maxValue);

            return make_int3(r, g, b);
        }

        // Convert HSL hue to RGB
        float hue2RGB(float t) {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
            if (t < 1.0f / 2.0f) return q;
            if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
            return p;
        }

    public:
        explicit ColorGenerator(uint32_t allColors) : numberOfColors(allColors) { }

        // Get color string in Chimera format for element at index
        std::string getNextColor(uint32_t index) {
            if (numberOfColors == 0) {
                return "";
            }
            int3 rgb = getNextRGBColor(index);

            std::ostringstream buf;
            buf << "#" << std::hex
                << std::setfill('0') << std::setw(4) << rgb.x
                << std::setfill('0') << std::setw(4) << rgb.y
                << std::setfill('0') << std::setw(4) << rgb.z;

            return buf.str();
        }
};

#endif //__COLOR_GENERATOR_H__
