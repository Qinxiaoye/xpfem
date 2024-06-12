import numpy as np
import struct


class fileBinaClass:
    # 写入矩阵
    def write_matrix(self,fileAddress,data,matrixType):
        # 判断矩阵维数
        ndim = data.ndim
        if ndim == 2:
            row,col = data.shape
            # 写入矩阵维数
            fileAddress.write(struct.pack('3q',ndim,row,col))
            if matrixType == 'int':
                dtype = 'q'
            elif matrixType == 'float':
                dtype = 'd'
            else:
                return 0
                    
            for n in range(0,row):
                for m in range(0,col):
                    s = struct.pack(dtype,data[n,m])
                    fileAddress.write(s)
        else:
            row, = data.shape
            # 写入矩阵维数
            fileAddress.write(struct.pack('2q',ndim,row))
            if matrixType == 'int':
                dtype = 'q'
            elif matrixType == 'float':
                dtype = 'd'
            else:
                return 0
                    
            for n in range(0,row):
                s = struct.pack(dtype,data[n])
                fileAddress.write(s)
        
        return 1
    # 读矩阵
    def read_matrix(self,fileAddress,matrixType):
        ndim, = struct.unpack('q',fileAddress.read(8))
        if ndim == 2:
            row,col = struct.unpack('2q',fileAddress.read(2*8))
            if matrixType == 'int':
                dtype = 'q'
                maDtype = np.int64
            elif matrixType == 'float':
                dtype = 'd'
                maDtype = np.float64
            else:
                return 0
            data = np.zeros((row,col),dtype=maDtype)
            for n in range(0,row):
                for m in range(0,col):
                    data[n,m] = struct.unpack(dtype,fileAddress.read(8))[0]
        else:
            row, = struct.unpack('q',fileAddress.read(8))
            if matrixType == 'int':
                dtype = 'q'
                maDtype = np.int64
            elif matrixType == 'float':
                dtype = 'd'
                maDtype = np.float64
            else:
                return 0
            data = np.zeros(row,dtype=maDtype)
            for n in range(0,row):
                data[n] = struct.unpack(dtype,fileAddress.read(8))[0]
        return data        

    # 写字符串
    def write_string(self,fileAddress,string):
        # length = len(string)
        s = bytes(string,encoding="utf-8")
        fileAddress.write(struct.pack("I%ds"%(len(s),),len(s),s))
        return 1
    
    def read_string(self,fileAddress):
        length, = struct.unpack('I',fileAddress.read(4))
        string = struct.unpack('%ds'%length,fileAddress.read(length))
        return string[0].decode()