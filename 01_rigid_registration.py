import SimpleITK as sitk
import os
import numpy as np

def extract_2d_slice(image, slice_index=None):
    """从3D图像中提取中间切片"""
    size = image.GetSize()
    if slice_index is None:
        slice_index = size[2] // 2  # 默认取中间切片
    
    extractor = sitk.ExtractImageFilter()
    extraction_size = list(size)
    extraction_size[2] = 0  # 将第三维度设为0表示提取2D切片
    
    extraction_index = [0, 0, slice_index]
    
    extractor.SetSize(extraction_size)
    extractor.SetIndex(extraction_index)
    
    return extractor.Execute(image)

def process_mask(mask):
    """处理mask，只将值为200和500的像素设置为1，其他值设置为0"""
    # 创建两个二值图像，分别标记200和500的位置
    mask_200 = sitk.Equal(mask, 200)
    mask_500 = sitk.Equal(mask, 500)
    
    # 将两个mask合并（逻辑或运算）
    combined_mask = sitk.Or(mask_200, mask_500)
    
    # 转换为整数类型（0和1）
    return sitk.Cast(combined_mask, sitk.sitkUInt8)

def normalize_image(image):
    """将图像归一化到0到255"""
    image_array = sitk.GetArrayFromImage(image)
    min_val = image_array.min()
    max_val = image_array.max()
    
    # 避免除以零
    if max_val - min_val == 0:
        return image
    
    # 归一化到0-255
    normalized_array = 255 * (image_array - min_val) / (max_val - min_val)
    normalized_image = sitk.GetImageFromArray(normalized_array.astype(np.float32))  # 保持为浮点类型
    normalized_image.CopyInformation(image)  # 复制图像信息
    return normalized_image

def perform_rigid_registration(fixed_image, moving_image, fixed_mask=None, moving_mask=None):
    """执行刚性配准，只使用mask进行配准"""
    if fixed_mask is None or moving_mask is None:
        raise ValueError("Masks are required for rigid registration")
    
    # 将mask转换为浮点类型
    fixed_mask = sitk.Cast(fixed_mask, sitk.sitkFloat32)
    moving_mask = sitk.Cast(moving_mask, sitk.sitkFloat32)
        
    registration_method = sitk.ImageRegistrationMethod()

    # 设置相似度度量（直接使用mask进行配准）
    registration_method.SetMetricAsMeanSquares()
    
    # 设置刚性变换
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_mask, 
        moving_mask, 
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS)  # 使用MOMENTS而不是GEOMETRY
    
    registration_method.SetInitialTransform(initial_transform)

    # 设置优化器参数（更保守的设置）
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.1,  # 降低学习率
        numberOfIterations=200,  # 增加迭代次数
        convergenceMinimumValue=1e-5,
        convergenceWindowSize=5,
        estimateLearningRate=registration_method.EachIteration)  # 自动估计学习率

    # 设置多分辨率策略（更细致的层级）
    registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # 设置插值方法
    registration_method.SetInterpolator(sitk.sitkLinear)

    print("Starting rigid registration using masks...")
    try:
        return registration_method.Execute(fixed_mask, moving_mask)
    except Exception as e:
        print(f"Registration failed: {str(e)}")
        print("Using initial transform as fallback")
        return initial_transform


def register_images(fixed_image_path, moving_image_path, fixed_mask_path=None, moving_mask_path=None, output_path=None, slice_index=None):
    # 读取图像
    fixed_image_3d = sitk.ReadImage(fixed_image_path)
    moving_image_3d = sitk.ReadImage(moving_image_path)
    
    # 提取2D切片
    fixed_image = extract_2d_slice(fixed_image_3d, slice_index)
    moving_image = extract_2d_slice(moving_image_3d, slice_index)
    
    # 确保图像方向一致
    fixed_direction = [1, 0, 0, 1]  # 2D图像方向矩阵
    moving_direction = [1, 0, 0, 1]
    fixed_image.SetDirection(fixed_direction)
    moving_image.SetDirection(moving_direction)
    
    # 读取并提取mask的2D切片（如果提供）
    fixed_mask = None
    moving_mask = None
    if fixed_mask_path and moving_mask_path:
        fixed_mask_3d = sitk.ReadImage(fixed_mask_path)
        moving_mask_3d = sitk.ReadImage(moving_mask_path)
        
        # 提取mask切片并进行二值化处理
        fixed_mask = extract_2d_slice(fixed_mask_3d, slice_index)
        moving_mask = extract_2d_slice(moving_mask_3d, slice_index)
        
        # 二值化处理（只保留200和500的区域）
        fixed_mask = process_mask(fixed_mask)
        moving_mask = process_mask(moving_mask)
        
        # 设置mask的方向
        fixed_mask.SetDirection(fixed_direction)
        moving_mask.SetDirection(moving_direction)
        
        # 保存处理后的mask用于验证
        if output_path:
            base, ext = os.path.splitext(output_path)
            if ext == '.gz':
                base, _ = os.path.splitext(base)
            sitk.WriteImage(fixed_mask, f"{base}_fixed_mask_slice{slice_index}.nii{ext}")
            sitk.WriteImage(moving_mask, f"{base}_moving_mask_slice{slice_index}.nii{ext}")
    
    # 保存固定图像切片
    if output_path:
        base, ext = os.path.splitext(output_path)
        if ext == '.gz':
            base, _ = os.path.splitext(base)
        sitk.WriteImage(fixed_image, f"{base}_fixed_slice{slice_index}.nii{ext}")
        sitk.WriteImage(moving_image, f"{base}_moving_slice{slice_index}.nii{ext}")

    # 1. 先进行刚性配准
    rigid_transform = perform_rigid_registration(fixed_image, moving_image, fixed_mask, moving_mask)
    
    # 保存刚性配准结果
    if output_path:
        base, ext = os.path.splitext(output_path)
        if ext == '.gz':
            base, _ = os.path.splitext(base)
        
        # 应用刚性变换
        rigid_resampler = sitk.ResampleImageFilter()
        rigid_resampler.SetReferenceImage(fixed_image)
        rigid_resampler.SetInterpolator(sitk.sitkLinear)
        rigid_resampler.SetTransform(rigid_transform)
        rigid_result = rigid_resampler.Execute(moving_image)
        
        # 保存刚性配准结果
        sitk.WriteImage(rigid_result, f"{base}_rigid_slice{slice_index}.nii{ext}")
    
    # 重采样移动mask
    if moving_mask is not None:
        mask_resampler = sitk.ResampleImageFilter()
        mask_resampler.SetReferenceImage(fixed_image)
        mask_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        mask_resampler.SetTransform(rigid_transform)
        transformed_moving_mask = mask_resampler.Execute(moving_mask)
        
        # 保存变换后的mask
        sitk.WriteImage(transformed_moving_mask, f"{base}_rigid_mask_slice{slice_index}.nii{ext}")
        # print(f"Transformed moving mask saved to: {output_mask_path}")


    return rigid_result, rigid_transform

def main():
    # 设置路径
    base_dir = "./data"
    out_dir = "./output"
    fixed_image_path = os.path.join(base_dir, "sub2092_lge.nii.gz")
    fixed_mask_path = os.path.join(base_dir, "sub2092_lge_mask.nii.gz")
    
    # 要配准的图像列表
    moving_images = [
        ("sub2092_t2mapt2.nii.gz", "sub2092_t2mapt2_mask.nii.gz"),
        ("sub2092_nativet1mapt1.nii.gz", "sub2092_nativet1mapt1_mask.nii.gz")
    ]

    # 读取固定图像以获取切片数量
    fixed_image = sitk.ReadImage(fixed_image_path)
    num_slices = fixed_image.GetSize()[2]
    slice_index = num_slices // 2  # 默认使用中间切片
    print(f"\nTotal number of slices: {num_slices}")
    print(f"Using middle slice: {slice_index}")

    for moving_image, moving_mask in moving_images:
        moving_image_path = os.path.join(base_dir, moving_image)
        moving_mask_path = os.path.join(base_dir, moving_mask)
        
        # 创建输出文件名
        output_filename = f"registered_{moving_image}"
        output_path = os.path.join(out_dir, output_filename)
        
        print(f"\nRegistering {moving_image} to LGE image...")
        try:
            registered_image, transform = register_images(
                fixed_image_path,
                moving_image_path,
                fixed_mask_path,
                moving_mask_path,
                output_path,
                slice_index
            )
            print(f"Successfully registered {moving_image}")
        except Exception as e:
            print(f"Error registering {moving_image}: {str(e)}")

if __name__ == "__main__":
    main()
