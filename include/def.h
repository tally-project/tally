#ifndef DEF_H
#define DEF_H

struct __align__(8) fatBinaryHeader
{
    unsigned int           magic;
    unsigned short         version;
    unsigned short         headerSize;
    unsigned long long int fatSize;
};

#endif // DEF_H