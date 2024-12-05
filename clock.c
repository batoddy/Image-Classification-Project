#include <stdio.h>
#include <time.h>

int main()
{
    clock_t ticks = clock();
    printf("Anlık tick değeri: %ld\n", ticks);
    for (int i = 0; i < 100000000; i++)
    {
    }
    for (int i = 0; i < 100000000; i++)
    {
    }
    for (int i = 0; i < 100000000; i++)
    {
    }
    for (int i = 0; i < 100000000; i++)
    {
    }

    for (int i = 0; i < 100000000; i++)
    {
    }
    for (int i = 0; i < 100000000; i++)
    {
    }
    for (int i = 0; i < 100000000; i++)
    {
    }

    ticks = clock();
    printf("Anlık tick değeri: %ld\n", ticks);
    return 0;
}
