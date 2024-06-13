import ctypes
import sys
import os

import multiprocessing 
import time 

def init_multyx_types(path, multyxlib):
    os.chdir(path)
    # multyx.dll is a shared library that exposes a few standard C functions.
    # ctypes is a python package for calling C functions.
    # Load multyx.dll using the python ctypes.cdll.LoadLibrary() function:
    
    # In order to tell python how to call each function in the DLL,
    # specify the python-C interface of thes functions:
    #
    # void* OpenMultyxSession(
    #   char* SessionFileName,
    #   MsgCallbackFunctionPointer ptr_infocallback,
    #   MsgCallbackFunctionPointer ptr_errorcallback,
    #   MsgCallbackFunctionPointer ptr_warningcallback
    # );
    multyxlib.OpenMultyxSession.restype=ctypes.c_void_p
    multyxlib.OpenMultyxSession.argtypes=[
    ctypes.c_char_p,
    ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p),
    ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p),
    ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)
    ]
    #
    #  Interface for setting values of multyx variables.
    # int SetValueFloatingPointVariable(void* SessionHandle,char* InputSpecifier,double Value);
    multyxlib.SetValueFloatingPointVariable.restype=ctypes.c_int
    multyxlib.SetValueFloatingPointVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_double]
    # int SetValueBoolVariable(void* SessionHandle,char* InputSpecifier,bool Value);
    multyxlib.SetValueBoolVariable.restype=ctypes.c_int
    multyxlib.SetValueBoolVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_bool]
    # int SetValueSwitchVariable(void* SessionHandle,char* InputSpecifier,int Value);
    multyxlib.SetValueSwitchVariable.restype=ctypes.c_int
    multyxlib.SetValueSwitchVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_int]
    # int SetValueIntegerVariable(void* SessionHandle,char* InputSpecifier,int Value);
    multyxlib.SetValueIntegerVariable.restype=ctypes.c_int
    multyxlib.SetValueIntegerVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_int]
    # int SetValueStringVariable(void* SessionHandle,char* InputSpecifier,char* Value);
    multyxlib.SetValueStringVariable.restype=ctypes.c_int
    multyxlib.SetValueStringVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_char_p]
    # int ExecuteScript(void* SessionHandle,char* Script,int ShowOutput);
    multyxlib.ExecuteScript.restype=ctypes.c_int
    multyxlib.ExecuteScript.argtypes=[ctypes.c_void_p,ctypes.c_char_p]
    #  Interface for getting values of multyx variables.
    # int GetValueFloatingPointTaggedItem(void* SessionHandle,char* OutputSpecifier,double* Value);
    multyxlib.GetValueFloatingPointTaggedItem.restype=ctypes.c_int
    multyxlib.GetValueFloatingPointTaggedItem.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_double)]
    # int GetValueBoolVariable(void* SessionHandle,char* OuputSpecifier,bool* Value);
    multyxlib.GetValueBoolVariable.restype=ctypes.c_int
    multyxlib.GetValueBoolVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_bool)]
    # int GetValueSwitchVariable(void* SessionHandle,char* OuputSpecifier,int* Value);
    multyxlib.GetValueSwitchVariable.restype=ctypes.c_int
    multyxlib.GetValueSwitchVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_int)]
    # int GetValueIntegerVariable(void* SessionHandle,char* OuputSpecifier,int* Value);
    multyxlib.GetValueIntegerVariable.restype=ctypes.c_int
    multyxlib.GetValueIntegerVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_int)]
    multyxlib.GetValueStringVariable.restype=ctypes.c_int
    # Call GetValueStringVariable() like this:
    # bufsize=1024
    # buf= (ctypes.c_char * buf_size)()
    # pbuf= (ctypes.POINTER(ctypes.c_char) * 1)(buf)
    # retval=multyxlib.GetValueStringVariable(SessionHandle,b"DESCRIPTION",pbuf,bufsize)
    multyxlib.GetValueStringVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.POINTER(ctypes.c_char)),ctypes.c_int]
    multyxlib.CloseMultyxSession.restype=ctypes.c_int
    multyxlib.CloseMultyxSession.argtypes=[ctypes.c_void_p]
    #
    # Default Null callback. Use this if you do not wish to process
    # Informational, Error, or Warning messages:
    # NullMsgCallback=ctypes.cast(None,ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p))    
    #
    # ShowOutput=1 if you want to see the output from multyx, =0 otherwise
    ShowOutput=ctypes.c_int(0) #Don't show output
    return multyxlib

# Take a message as parameter, and return a bool DoAboort
@ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)
def InfoCallBack(msg):
    #print(msg.decode("utf-8"))
    # Do not abort session:
    return ctypes.c_bool(False)

# Take a message as parameter, and return a bool DoAboort
@ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)
def ErrorCallBack(msg):
    print(msg.decode("utf-8"))
    # Do not abort when an error message is received:
    return ctypes.c_bool(False)

# Take a message as parameter, and return a bool DoAboort
@ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)
def WarningCallBack(msg):
    print(msg.decode("utf-8"))
    # Do not abort session:
    return ctypes.c_bool(False)

def init_multyx_session(path, ses_file, multyxlib):
    #sys.path.insert(0,'/opt/ansol/calyx')
    
    # Start a session by passing it the name of the session file to read.
    # It will acquire the license, open the session file and wait.
    # It will return a SessionHandle of type ctypes.c_voidp, which is needed
    # for all subsequent calls to the opened session.
    SessionFileName=bytes(ses_file, encoding='utf-8')
    SessionHandle=multyxlib.OpenMultyxSession(SessionFileName, InfoCallBack, ErrorCallBack, WarningCallBack)
    if SessionHandle==ctypes.c_voidp(None):
        print("Failed to start session")
        return 1
    return (multyxlib, SessionHandle)


class Process(multiprocessing.Process): 
    def __init__(self, id, path, ses_file): 
        super(Process, self).__init__(target=Interface, args=(path, ses_file))
        self.id = id
        


class Interface:
    def __init__(self, path, ses_file, library) -> None:
        self.path = path
        self.ses_file = ses_file
        
        self.interface, self.ses_handle = init_multyx_session(path, ses_file, library)

    def runScript(self, script):
        retval = self.interface.ExecuteScript(self.ses_handle, script)
        if retval==0:
            print("Failed to run the script")
        return

    def __exit__(self, exc_type, exc_value, traceback):
        retval=self.interface.CloseMultyxSession(self.ses_handle) # release the license
        if retval==0:
            print("Failed to release session")

        del(self.interface) # release the dll library

def main():
    path = r'C:\Users\egrab\Desktop\T3D_test\load_1'
    ses_file = 'T3D_sim.ses'
    process1 = multiprocessing.Process(target = Interface, args=(path, ses_file))
    process1.start()
    path = r'C:\Users\egrab\Desktop\T3D_spur\load_1'
    ses_file = 'T3D_spur_sim.ses'
    process2 = multiprocessing.Process(target = Interface, args=(path, ses_file))
    process2.start()

if __name__ == '__main__':
    main()