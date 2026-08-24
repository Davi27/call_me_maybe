import sys
import json

def main() -> None:
    if len(sys.argv) != 4:
        print("Wrong number of arguments")
        return

    try:
        with open(sys.argv[1], "r") as f:
            funcs = json.load(f)
    except FileNotFoundError:
        print("File not found", sys.argv[1])
        return
    except ValueError:
        print(sys.argv[1], "contains the incorrect format")
        return

    try:
        with open(sys.argv[2], "r") as f:
            inpt = json.load(f)
    except FileNotFoundError:
        print("File not found", sys.argv[2])
        return
    except ValueError:
        print(sys.argv[2], "contains the incorrect format")
        return

    with open(sys.argv[3], "w") as f:
        json.dump(inpt, f, indent=2)



if __name__ == "__main__":
    main()
