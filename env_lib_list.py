

if __name__=="__main__":
    print(f"Start checking environment packages:")
    try:
        import numpy
        print(f"numpy={numpy.__version__}")
    except:
        print(f"numpy package is not installed in the environment")

    try:
        import matplotlib

        print(f"matplotlib={matplotlib.__version__}")
    except:
        print(f"matplotlib package is not installed in the environment")

    try:
        import scipy

        print(f"scipy={scipy.__version__}")
    except:
        print(f"scipy package is not installed in the environment")

    try:
        import tensorflow

        print(f"tensorflow={tensorflow.__version__}")
    except:
        print(f"tensorflow package is not installed in the environment")

    try:
        import pandas

        print(f"pandas={pandas.__version__}")
    except:
        print(f"pandas package is not installed in the environment")

    try:
        import sklearn

        print(f"sklearn={sklearn.__version__}")
    except:
        print(f"sklearn package is not installed in the environment")


    print(f"Finished checking")


