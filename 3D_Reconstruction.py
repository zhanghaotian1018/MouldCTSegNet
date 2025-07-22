import vtk
import os
import glob
import argparse

from vtkmodules.vtkInteractionWidgets import vtkButtonWidget


def parser_init():
    parser = argparse.ArgumentParser(description='3D Reconstruction')
    parser.add_argument('--sample_folder', type=str,
                        default=".//datasets//H351-1-0001", help='The path to the folder where original images are stored')
    parser.add_argument('--mask_folder', type=str,
                        default=".//datasets//H351-1-0001_pred", help='The path to the folder where binary mask images are stored')
    parser.add_argument('--output_folder', type=str,
                        default=".//datasets//H351-1-0001_output", help='The path to the folder where different target region images will be saved')
    parser.add_argument('--start_index', type=int, default=0,
                        help='The start index of the image to be processed')
    parser.add_argument('--end_index', type=int, default=1000,
                        help='The end index of the image to be processed')
    return parser

def ReadImages(files, format):
    if format.lower()=='png':
        reader = vtk.vtkPNGReader()
        image3D = vtk.vtkImageAppend()
        image3D.SetAppendAxis(2)

        for f in files:
            reader.SetFileName(f)
            reader.Update()
            t_img = vtk.vtkImageData()
            t_img.DeepCopy(reader.GetOutput())
            image3D.AddInputData(t_img)
    else:
        reader = vtk.vtkBMPReader()
        image3D = vtk.vtkImageAppend()
        image3D.SetAppendAxis(2)

        for f in files:
            reader.SetFileName(f)
            reader.Update()
            t_img = vtk.vtkImageData()
            t_img.DeepCopy(reader.GetOutput())
            image3D.AddInputData(t_img)

    image3D.Update()

    return image3D.GetOutput()


def Load_Files(args):
    # Set image path and loaded in sort
    start_index = args.start_index
    end_index = args.end_index
    bright_png_folder = os.path.join(args.output_folder, 'bright_part')
    dark_png_folder = os.path.join(args.output_folder, 'dark_part')

    bright_png_files = glob.glob(bright_png_folder + os.sep + '*.png')[start_index:end_index]
    bright_png_files.sort()
    dark_png_files = glob.glob(dark_png_folder + os.sep + '*.png')[start_index:end_index]
    dark_png_files.sort()

    dark_data_list = ReadImages(dark_png_files, 'png')
    dark_data_list.SetSpacing([1, 1, 1])

    bright_data_list = ReadImages(bright_png_files, 'png')
    bright_data_list.SetSpacing([1, 1, 1])

    # Set original image and binary mask path and loaded in sort
    ori_png_folder = args.sample_folder
    ori_png_files = glob.glob(ori_png_folder + os.sep + '*.png')[start_index:end_index]
    ori_png_files.sort()
    mask_png_folder = args.mask_folder
    mask_png_files = glob.glob(mask_png_folder + os.sep + '*.png')[start_index:end_index]
    mask_png_files.sort()

    ori_data_list = ReadImages(ori_png_files, 'png')
    ori_data_list.SetSpacing([1, 1, 1])

    mask_data_list = ReadImages(mask_png_files, 'png')
    mask_data_list.SetSpacing([1, 1, 1])

    return bright_data_list, dark_data_list, ori_data_list, mask_data_list


def VTK_Init():
    # Create a vtkRenderer
    renderer_left = vtk.vtkRenderer()
    renderer_left.SetBackground(0.8, 0.8, 0.8)
    renderer_left.SetViewport(0.0, 0.0, 0.5, 1.0)

    renderer_right = vtk.vtkRenderer()
    renderer_right.SetBackground(0.6, 0.6, 0.6)
    renderer_right.SetViewport(0.5, 0.0, 1.0, 1.0)

    global camera_left_ori_pos
    camera_left_ori_pos = (927.0, 1155.5, 6316.262193989753)

    # Create a vtkRenderWindow
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1200, 600)
    render_window.SetWindowName('3D Visualization')
    render_window.AddRenderer(renderer_left)
    render_window.AddRenderer(renderer_right)

    # Create a vtkRenderWindowInteractor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    style = vtk.vtkInteractorStyleTrackballCamera()
    render_window_interactor.SetInteractorStyle(style)

    # Create a vtkNamedColors
    colors = vtk.vtkNamedColors()

    return renderer_left, renderer_right, render_window, render_window_interactor, colors


def Outline_Widget(image_data_list):
    outlineData = vtk.vtkOutlineFilter()
    outlineData.SetInputData(image_data_list)
    outline_mapper = vtk.vtkPolyDataMapper()
    outline_mapper.SetInputConnection(outlineData.GetOutputPort())

    outline = vtk.vtkActor()  # LineActor
    outline.SetMapper(outline_mapper)
    outline.GetProperty().SetColor(0, 0, 0)
    renderer_left.AddActor(outline)


def Show_All(render_window, render_window_interactor):
    ori_widget.On()
    render_window.Render()

    render_window_interactor.Initialize()
    render_window_interactor.Start()


def create_button(render_window_interactor):
    # Define the button's image
    pngReader = vtk.vtkPNGReader()
    pngReader.SetFileName("./utils/Button.png")
    pngReader.Update()
    image = pngReader.GetOutput()

    # Define the button's size and position
    sz = 150.0
    bds = [0, 0, 0, 0, 0, 0]
    bds[0] = 4096
    bds[1] = bds[0] + sz / 2
    bds[2] = 0
    bds[3] = bds[2] + sz / 2

    # Create the button's representation
    buttonRepresentation = vtk.vtkTexturedButtonRepresentation2D()
    buttonRepresentation.SetNumberOfStates(1)
    buttonRepresentation.SetButtonTexture(0, image)
    buttonRepresentation.PlaceWidget(bds)
    '''
    PlaceWidget() - given a bounding box (xmin,xmax,ymin,ymax,zmin,zmax), place
                    the widget inside of it.
    '''
    # create the button widget and add it to the interactor
    global mbutton, mbutton_state
    mbutton_state = 0
    mbutton = vtkButtonWidget()
    mbutton.SetInteractor(render_window_interactor)
    mbutton.SetRepresentation(buttonRepresentation)
    mbutton.AddObserver("StateChangedEvent", my_button_callback)
    mbutton.SetEnabled(1)


def my_button_callback(obj, event):
    global mbutton_state, bright_volume, dark_volume
    mbutton_state += 1
    if mbutton_state == 3:
        mbutton_state = 0
    '''
    0 -- Display All
    1 -- Display Bright Part
    2 -- Display Dark Part
    '''
    if mbutton_state == 0:
        print('Display All')
        renderer_left.RemoveVolume(dark_volume)
        renderer_left.RemoveVolume(bright_volume)
        renderer_left.AddVolume(ori_volume)
    elif mbutton_state == 1:
        print('Display Bright Part')
        renderer_left.RemoveVolume(ori_volume)
        renderer_left.RemoveVolume(dark_volume)
        renderer_left.AddVolume(bright_volume)
    else:
        print('Display Dark Part')
        renderer_left.RemoveVolume(ori_volume)
        renderer_left.RemoveVolume(bright_volume)
        renderer_left.AddVolume(dark_volume)


def show_bright_and_dark_part(bright_data_output, dark_data_output, ori_data_output):
    # Bright and Dark Volume Data Filter
    bright_volume_data_filter = vtk.vtkImageGaussianSmooth()
    bright_volume_data_filter.SetInputData(bright_data_output)
    bright_volume_data_filter.SetStandardDeviation(4.0)  # change the smooth level
    bright_volume_data_filter.Update()

    dark_volume_data_filter = vtk.vtkImageGaussianSmooth()
    dark_volume_data_filter.SetInputData(dark_data_output)
    dark_volume_data_filter.SetStandardDeviation(4.0)  # change the smooth level
    dark_volume_data_filter.Update()

    # Create a vtkGPUVolumeRayCastMapper object and set the input data
    bright_volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
    bright_volume_mapper.SetInputConnection(bright_volume_data_filter.GetOutputPort())

    dark_volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
    dark_volume_mapper.SetInputConnection(dark_volume_data_filter.GetOutputPort())

    ori_volume_mapeer = vtk.vtkGPUVolumeRayCastMapper()
    ori_volume_mapeer.SetInputData(ori_data_output)

    # Set color and opacity transfer functions
    bd_color_func = vtk.vtkColorTransferFunction()
    bd_color_func.AddRGBPoint(40, 0.0, 0.0, 0.0)
    bd_color_func.AddRGBPoint(200, 1.0, 1.0, 1.0)
    bd_opacity_func = vtk.vtkPiecewiseFunction()
    bd_opacity_func.AddPoint(40, 0.0)
    bd_opacity_func.AddPoint(255, 1.0)

    # Set the volume properties
    bright_volume_property = vtk.vtkVolumeProperty()
    bright_volume_property.SetColor(bd_color_func)
    bright_volume_property.SetScalarOpacity(bd_opacity_func)
    bright_volume_property.SetInterpolationTypeToLinear()

    dark_volume_property = vtk.vtkVolumeProperty()
    dark_volume_property.SetColor(bd_color_func)
    dark_volume_property.SetScalarOpacity(bd_opacity_func)
    dark_volume_property.SetInterpolationTypeToLinear()

    # Set color and opacity transfer functions
    color_func = vtk.vtkColorTransferFunction()
    color_func.AddRGBPoint(20, 0.0, 0.0, 0.0)
    color_func.AddRGBPoint(255, 1.0, 1.0, 1.0)
    opacity_func = vtk.vtkPiecewiseFunction()
    opacity_func.AddPoint(20, 0.0)
    opacity_func.AddPoint(255, 1.0)

    ori_volume_property = vtk.vtkVolumeProperty()
    ori_volume_property.SetColor(color_func)
    ori_volume_property.SetScalarOpacity(opacity_func)
    ori_volume_property.SetInterpolationTypeToLinear()

    # Create a vtkVolume object and set the mapper and property
    global bright_volume, dark_volume, ori_volume
    bright_volume = vtk.vtkVolume()
    bright_volume.SetMapper(bright_volume_mapper)
    bright_volume.SetProperty(bright_volume_property)

    dark_volume = vtk.vtkVolume()
    dark_volume.SetMapper(dark_volume_mapper)
    dark_volume.SetProperty(dark_volume_property)

    ori_volume = vtk.vtkVolume()
    ori_volume.SetMapper(ori_volume_mapeer)
    ori_volume.SetProperty(ori_volume_property)

    # Add the volume to the renderer
    renderer_left.AddVolume(ori_volume)


def create_TF_lut(colorful_flag):
    '''
    colorful_flag: 0 = Ori gray image， 1 = Mask colorful image
    '''
    # create a vtkLookupTable object
    lut = vtk.vtkLookupTable()
    # Set the number of colors to 300
    lut.SetNumberOfColors(300)
    # Set the range of the scalar values to map from 0 to 299
    lut.SetTableRange(0, 299)
    # Set lut on
    lut.Build()

    if colorful_flag == 0:
        for i in range(1):
            lut.SetTableValue(i, [0.0, 0.0, 0.0, 0.0])

        for i in range(1, 299):
            lut.SetTableValue(i, [i/(299-1), i/(299-1), i/(299-1), 0.8])
    else:
    # Set colorful lut
        for i in range(40):
            lut.SetTableValue(i, [0.0, 0.0, 0.0, 0.0])

        for i in range(40, 130):
            lut.SetTableValue(i, [0.0, 0.5, 0.0, 0.5])

        for i in range(130, 299):
            lut.SetTableValue(i, [0.5, 0.0, 0.0, 0.5])

    return lut


def display_ori_image_on_right_viewport():
    ori_plane = vtk.vtkPlaneSource()
    ori_plane_poly_data_mapper = vtk.vtkPolyDataMapper()
    ori_plane_poly_data_mapper.SetInputConnection(ori_plane.GetOutputPort())

    ori_lut = create_TF_lut(colorful_flag=0)    # to display more clear original image
    ori_plane_texture = vtk.vtkTexture()
    ori_plane_texture.SetInputConnection(ori_image_reslice.GetOutputPort())
    ori_plane_texture.SetLookupTable(ori_lut)
    ori_plane_texture.SetColorModeToMapScalars()    # to display more clear original image

    ori_plane_actor = vtk.vtkActor()
    ori_plane_actor.SetMapper(ori_plane_poly_data_mapper)
    ori_plane_actor.SetTexture(ori_plane_texture)

    renderer_right.AddActor(ori_plane_actor)


def display_mask_image_on_right_viewport():
    mask_image_plane = vtk.vtkPlaneSource()
    mask_image_mapper = vtk.vtkPolyDataMapper()
    mask_image_mapper.SetInputConnection(mask_image_plane.GetOutputPort())

    mask_lut = create_TF_lut(colorful_flag=1)
    mask_color_map = vtk.vtkImageMapToColors()
    mask_color_map.SetInputConnection(mask_image_reslice.GetOutputPort())
    mask_color_map.SetLookupTable(mask_lut)
    mask_color_map.Update()
    mask_texture = vtk.vtkTexture()
    mask_texture.SetInputConnection(mask_color_map.GetOutputPort())

    mask_image_actor = vtk.vtkActor()
    mask_image_actor.SetMapper(mask_image_mapper)
    mask_image_actor.SetTexture(mask_texture)
    mask_image_actor.GetProperty().SetOpacity(0.5)

    renderer_right.AddActor(mask_image_actor)


def main():
    parser = parser_init()
    args = parser.parse_args()
    # -------------------------------------------------------------
    global renderer_left, renderer_right
    # VTK Init
    (renderer_left,
     renderer_right,
     render_window,
     render_window_interactor,
     colors) = VTK_Init()

    # -------------------------------------------------------------
    # Get Bright and Dark 3d image data's Output()
    global bright_data_output, dark_data_output
    bright_data_output, dark_data_output, ori_data_output, mask_data_output = Load_Files(args)

    # -------------------------------------------------------------
    # Display bright and dark part
    show_bright_and_dark_part(bright_data_output, dark_data_output, ori_data_output)

    # widget to display original image
    global ori_widget
    ori_widget = vtk.vtkImagePlaneWidget()
    ori_widget.TextureVisibilityOff()
    ori_widget.SetInputData(ori_data_output)
    ori_widget.SetPlaneOrientationToZAxes()
    ori_widget.SetInteractor(render_window_interactor)
    ori_widget.SetSliceIndex(1)

    # -------------------------------------------------------------
    # load mask image and reslice it to fit the original image
    global mask_image_reslice, ori_image_reslice
    mask_image_reslice = vtk.vtkImageReslice()
    mask_image_reslice.SetInputData(mask_data_output)
    mask_image_reslice.SetOutputDimensionality(2)
    mask_image_reslice.SetResliceAxes(ori_widget.GetResliceAxes())
    mask_image_reslice.SetOutputScalarType(vtk.VTK_UNSIGNED_CHAR)
    mask_image_reslice.Update()

    ori_image_reslice = vtk.vtkImageReslice()
    ori_image_reslice.SetInputData(ori_data_output)
    ori_image_reslice.SetOutputDimensionality(2)
    ori_image_reslice.SetResliceAxes(ori_widget.GetResliceAxes())
    ori_image_reslice.SetOutputScalarType(vtk.VTK_UNSIGNED_CHAR)
    ori_image_reslice.Update()

    # -------------------------------------------------------------
    # Add a plane in right viewport to display fusion segmentation result and original image
    display_ori_image_on_right_viewport()
    display_mask_image_on_right_viewport()

    # -------------------------------------------------------------
    # Add Button
    create_button(render_window_interactor)

    # -------------------------------------------------------------
    # Display All and Start Interactor
    Show_All(render_window, render_window_interactor)


if __name__ == '__main__':
    main()
