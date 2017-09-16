<Query Kind="Program" />

void Main()
{
    Random rnd = new Random(3);
    for (int i = 0; i < 1000; i++)
    {
        int config = rnd.Next(5) + 1;
        int value1 = rnd.Next(1000);
        int value2 = rnd.Next(1000);
        int result = 0;
        switch (config)
        {
            case 1:
                result = value1 > value2 ? 0 : 1;
                break;
            case 2:
                result = value1 < value2 ? 0 : 1;
                break;
            case 3:
                result = value1 < 500 ? 0 : 1;
                break;
            case 4:
                result = value1 < value2 - 100  ? 0 : 1;
                break;
            case 5:
                result = value1 < value2 + 100  ? 0 : 1;
                break;
            default:
                throw new Exception(config.ToString());
        }

       // Console.WriteLine($"{config},{value1},{value2}");
        Console.WriteLine($"{result}");
    }
}

// Define other methods and classes here
