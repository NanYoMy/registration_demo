import SimpleITK as sitk
import os

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

def perform_bspline_registration(fixed_image, moving_image, initial_transform, fixed_mask=None, moving_mask=None):
    """执行B样条FFD配准"""
    # 确保输入图像和mask为浮点类型
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    # 确保mask也是浮点类型
    if fixed_mask is not None:
        fixed_mask = sitk.Cast(fixed_mask, sitk.sitkFloat32)
    if moving_mask is not None:
        moving_mask = sitk.Cast(moving_mask, sitk.sitkFloat32)

    # 对移动图像进行平移，避免最小值为0
    moving_image += 1.0  # 将所有值加1，避免为0

    # 打印输入图像和mask的信息
    print(f"Fixed Image Size: {fixed_image.GetSize()}, Type: {fixed_image.GetPixelIDTypeAsString()}, Min: {sitk.GetArrayFromImage(fixed_image).min()}, Max: {sitk.GetArrayFromImage(fixed_image).max()}")
    print(f"Moving Image Size: {moving_image.GetSize()}, Type: {moving_image.GetPixelIDTypeAsString()}, Min: {sitk.GetArrayFromImage(moving_image).min()}, Max: {sitk.GetArrayFromImage(moving_image).max()}")
    if fixed_mask is not None:
        print(f"Fixed Mask Size: {fixed_mask.GetSize()}, Type: {fixed_mask.GetPixelIDTypeAsString()}, Min: {sitk.GetArrayFromImage(fixed_mask).min()}, Max: {sitk.GetArrayFromImage(fixed_mask).max()}")
    if moving_mask is not None:
        print(f"Moving Mask Size: {moving_mask.GetSize()}, Type: {moving_mask.GetPixelIDTypeAsString()}, Min: {sitk.GetArrayFromImage(moving_mask).min()}, Max: {sitk.GetArrayFromImage(moving_mask).max()}")

    registration_method = sitk.ImageRegistrationMethod()

    # 设置相似度度量为均方误差
    registration_method.SetMetricAsMeanSquares()
    
    # 如果提供了mask，设置mask
    if fixed_mask is not None and moving_mask is not None:
        registration_method.SetMetricFixedMask(fixed_mask)
        registration_method.SetMetricMovingMask(moving_mask)

    # 设置B样条变换
    mesh_size = [max(3, int(sz/8)) for sz in fixed_image.GetSize()[:2]]
    transform = sitk.BSplineTransformInitializer(fixed_image, mesh_size, order=3)
    
    # 将刚性变换的结果作为初始变换
    registration_method.SetInitialTransform(initial_transform, True)
    registration_method.SetMovingInitialTransform(transform)

    # 设置优化器
    registration_method.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=100,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e7)

    # 设置多分辨率策略
    registration_method.SetShrinkFactorsPerLevel([2, 1])
    registration_method.SetSmoothingSigmasPerLevel([1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # 设置插值方法
    registration_method.SetInterpolator(sitk.sitkLinear)

    # 禁用多线程
    registration_method.SetNumberOfThreads(1)

    print("Starting B-spline registration...")
    return registration_method.Execute(fixed_image, moving_image)

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
    
    # 2. 再进行B样条FFD配准
    final_transform = perform_bspline_registration(fixed_image, moving_image, rigid_transform, fixed_mask, moving_mask)

    # 重采样移动图像
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(final_transform)
    
    registered_image = resampler.Execute(moving_image)

    # 保存最终结果
    if output_path:
        base, ext = os.path.splitext(output_path)
        if ext == '.gz':
            base, _ = os.path.splitext(base)
        output_2d_path = f"{base}_slice{slice_index}.nii{ext}"
        
        sitk.WriteImage(registered_image, output_2d_path)
        print(f"Registered 2D image saved to: {output_2d_path}")
    
    return registered_image, final_transform

def main():
    # 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
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
        output_path = os.path.join(base_dir, output_filename)
        
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
