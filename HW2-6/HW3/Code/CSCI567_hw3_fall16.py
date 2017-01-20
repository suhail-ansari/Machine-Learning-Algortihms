import BVTradeOff
import BVTradeOff2

import os

import sys

def main():
    cont = True
    if not os.path.exists('./libsvm'):
        print "\n"
        print "You need to copy the libsvm folder in this directory (scripts root directory)"
        print "and rename the folder to 'libsvm' to run libsvm scripts."
        print "You can also install libsvm using the included install script, 'bash install_LIBSVM.sh' \n"
        print "The install script will install libsvm in this folder."
        print ">> Please make sure that the name of the folder containing libsvm is 'libsvm'"
        print "\n"
        cont = False
    
    if not cont: 
        r = raw_input("Continue? [y/n]") 
        if len(r) == 0 or r[0].lower == "y":
            run()
    else:
        run()
        

def run():
    BVTradeOff.main()
    BVTradeOff2.main()

    print "\nI have written a shell file to run all the cases for LIBSVM."
    print "Please make sure that the 'libsvm' folder is in this directory."
    print "You can try to run the shell files from python after the next prompt,"
    print "or you can the file from terminal using >> 'bash SVM.sh'"
    print "You can also try installing the libsvm using the install script I included by running >> 'bash install_LIBSVM.sh'"

    r = raw_input("\nDo you want to attempt to run the libsvm Shell File from python. [y/n]") 
    if len(r) == 0 or r[0] == "y":
        try:
            print "Generating LIBSVM formatted files from given files."
            import subprocess
            subprocess.call(["bash",  "./SVM.sh"], stdout=sys.stdout)
        except Exception as e:
            print e
            print "\nError: Unable to run the shell script from this python script."
            print "Please make sure the libsvm folder is in this directory, and run the file from"
            print "terminal using >> 'bash SVM.sh'"
    
    import SVM
    SVM.main()


if __name__ == "__main__":
    main()
