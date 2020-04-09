import numpy as np
from tqdm import tqdm

def read_pcd(fname):
    '''
    Read PCD data

    args: fname - Path to the PCD file
    returns: data - Nx6 matrix where each row is a point, with fields x y z rgb
                    imX imY. x, y, z are the 3D coordinates of the point, rgb is
                    the color of the point packed into a float (unpack using
                    unpackRGBFloat), imX and imY are the horizontal and vertical
                    pixel locations of the point in the original Kinect image.

    Matlab Author: Kevin Lai
    '''
    data = []
    version = 0
    width = 0
    height = 0
    points = 0

    final = []

    with open(fname, 'r') as f:
        lines = f.readlines()
        for l in lines:
            # clean line
            l = l.split(' ')
            l[-1] = l[-1].strip('\n')

            #
            if l[0].isalpha():
                if l[0] == "VERSION":
                    version = float(l[1])
                elif l[0] == 'WIDTH':
                    width = int(l[1])
                elif l[0] == 'HEIGHT':
                    height = int(l[1])
                elif l[0] == 'POINTS':
                    points = int(l[1])

            elif l[0] != '#':
                l = [float(i) for i in l]
                data.append(l)

    print('done with first bit')
    # for i in tqdm(range(points)):
    #     for j, d in enumerate(data):
    #         d.append(i)
    #         d.append(j)

    return np.array(data)

    # fid = open(fname, 'r')
    # is_binary = False
    #
    #
    # n_points = 0
    # n_dims = -1
    # line = []
    # format = []
    # header_length = 0
    # IS_NEW = True
    #
    #
    # while len(line) < 4 or not line[0:3] == 'DATA':
    #     line = fgetl(fid);
    #     if not line.isalpha():
    #         # end of file reached before finished parsing. No data
    #         data = np.zeros((0,6));
    #         return
    #
    #     header_length += len(line) + 1
    #
    #     if len(line) >= 4 and line[0:3] == 'TYPE':
    #         while not line == '':
    #
    #             # [t line] = strtok(line)
    #             if n_dims > -1 and t == 'F':
    #                 format.append('%f ')
    #             elif n_dims > -1 and t == 'U':
    #                 format.append('%d ')
    #             n_dims += 1
    #
    #     if len(line) >= 7 and line[0:6] == 'COLUMNS':
    #         IS_NEW = False
    #         while not line == '':
    #             splitted = line.split(' ')
    #             ig = splitted[0]
    #             line2 = splitted[1]
    #             format.append('%f ')
    #             n_dims += 1
    #
    #     if len(line) >= 6 and line[0:5] == 'POINTS':
    #
    #         splitted = line.split(' ')
    #         ig = splitted[0]
    #         l2 = splitted[1]
    #         n_points = int(l2)
    #
    #     if len(line) >= 4 and line[0:3] == 'DATA':
    #         if len(line) == 11 and line[5:10] == 'binary':
    #             is_binary = True
    #
    # format(end) = [];
    #
    # if is_binary:
    #    padding_length = 4096*np.ceil(header_length/4096);
    #    padding = read(fid,padding_length-header_length,'uint8');
    #
    # if is_binary and IS_NEW:
    #
    #     data = np.zeros((n_points, n_dims));
    #     format = regexp(format,' ','split');
    #     for i in n_points:
    #         for j in range(len(format)):
    #             if format[j] == '%d':
    #                 pt = read(fid,1,'uint32')
    #             else:
    #                 pt = read(fid,1,'float')
    #             data[i,j] = pt
    # elif is_binary and not IS_NEW:
    #     pts = read(fid,inf,'float')
    #     data = np.zeros((n_dims, n_points))
    #     data[:] = pts
    #     data = data
    # else:
    #    format.append('\n')
    #    C = textscan(fid,format)
    #
    #    data = cell2mat(C)
    #
    # fid.close()


'''

function data = readPcd(fname)
% Read PCD data
% fname - Path to the PCD file
% data - Nx6 matrix where each row is a point, with fields x y z rgb imX imY. x, y, z are the 3D coordinates of the point, rgb is the color of the point packed into a float (unpack using unpackRGBFloat), imX and imY are the horizontal and vertical pixel locations of the point in the original Kinect image.
%
% Author: Kevin Lai

fid = fopen(fname,'rt');

isBinary = false;
nPts = 0;
nDims = -1;
line = [];
format = [];
headerLength = 0;
IS_NEW = true;
while length(line) < 4 | ~strcmp(line(1:4),'DATA')
   line = fgetl(fid);
   if ~ischar(line)
      % end of file reached before finished parsing. No data
      data = zeros(0,6);
      return;
   end

   headerLength = headerLength + length(line) + 1;

   if length(line) >= 4 && strcmp(line(1:4),'TYPE') %COLUMNS
      while ~isempty(line)
         [t line] = strtok(line);
         if nDims > -1 && strcmp(t,'F')
            format = [format '%f '];
         elseif nDims > -1 && strcmp(t,'U')
            format = [format '%d '];
         end
         nDims = nDims+1;
      end
   end

   if length(line) >= 7 && strcmp(line(1:7),'COLUMNS')
      IS_NEW = false;
      while ~isempty(line)
         [ig line] = strtok(line);
         format = [format '%f '];
         nDims = nDims+1;
      end
   end

   if length(line) >= 6 && strcmp(line(1:6),'POINTS')
      [ig l2] = strtok(line);
      nPts = sscanf(l2,'%d');
   end

   if length(line) >= 4 && strcmp(line(1:4),'DATA')
      if length(line) == 11 && strcmp(line(6:11),'binary')
         isBinary = true;
      end
   end
end
format(end) = [];

if isBinary
   paddingLength = 4096*ceil(headerLength/4096);
   padding = fread(fid,paddingLength-headerLength,'uint8');
end

if isBinary && IS_NEW
   data = zeros(nPts,nDims);
   format = regexp(format,' ','split');
   for i=1:nPts
      for j=1:length(format)
         if strcmp(format{j},'%d')
            pt = fread(fid,1,'uint32');
         else
            pt = fread(fid,1,'float');
         end
         data(i,j) = pt;
      end
   end
elseif isBinary && ~IS_NEW
   pts = fread(fid,inf,'float');
   data = zeros(nDims,nPts);
   data(:) = pts;
   data = data';
else
   format = [format '\n'];
   C = textscan(fid,format);

   data = cell2mat(C);
end
fclose(fid);

'''

if __name__ == '__main__':
    print(read_pcd('../Data/data/0000000000.pcd'))
