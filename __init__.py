import os
import sys
import filecmp
import shutil
import __main__


python = sys.executable


extentions_folder = os.path.join(os.path.dirname(os.path.realpath(__main__.__file__)),
                                 "web" + os.sep + "extensions" + os.sep + "ZHO")
javascript_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mtb")

if not os.path.exists(extentions_folder):
    print('Making the "web\extensions\ZHO" folder')
    os.mkdir(extentions_folder)

result = filecmp.dircmp(javascript_folder, extentions_folder)

if result.left_only or result.diff_files:
    print('Update to javascripts files detected')
    file_list = list(result.left_only)
    file_list.extend(x for x in result.diff_files if x not in file_list)

    for file in file_list:
        print(f'Copying {file} to extensions folder')
        src_file = os.path.join(javascript_folder, file)
        dst_file = os.path.join(extentions_folder, file)
        if os.path.exists(dst_file):
            os.remove(dst_file)
        #print("disabled")
        shutil.copy(src_file, dst_file)


from .Zho_TextImage import NODE_CLASS_MAPPINGS  as NODE_CLASS_MAPPINGS_TI
from .Zho_RGB_Image import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_RGB
from .Zho_ImageComposite import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_IC
from .Zho_AlphaChanel import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_AC
from .Zho_TextImage_frame import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_TIF


# Combine the dictionaries
NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS_TI, **NODE_CLASS_MAPPINGS_RGB, **NODE_CLASS_MAPPINGS_IC, **NODE_CLASS_MAPPINGS_AC, **NODE_CLASS_MAPPINGS_TIF}


__all__ = ['NODE_CLASS_MAPPINGS']
