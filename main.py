import sys

def main() -> None:
    if len(sys.argv) != 4:
        print("Wrong number of arguments")
        return
    print("Arguments:", sys.argv[1:])

if __name__ == "__main__":
    main()
