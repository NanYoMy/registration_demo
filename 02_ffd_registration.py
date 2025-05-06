import SimpleITK as sitk
import os
import numpy as np


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


def perform_bspline_registration(fixed_image, moving_image, fixed_mask=None, moving_mask=None):
    """执行B样条FFD配准"""
    # 确保输入图像和mask为浮点类型
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    # 归一化图像到0-255
    fixed_image = normalize_image(fixed_image)
    moving_image = normalize_image(moving_image)
    
    # 确保mask也是浮点类型
    if fixed_mask is not None:
        fixed_mask = sitk.Cast(fixed_mask, sitk.sitkFloat32)
    if moving_mask is not None:
        moving_mask = sitk.Cast(moving_mask, sitk.sitkFloat32)

    registration_method = sitk.ImageRegistrationMethod()

    # 设置相似度度量为互信息
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    
    # 如果提供了mask，设置mask
    if fixed_mask is not None and moving_mask is not None:
        registration_method.SetMetricFixedMask(fixed_mask)
        registration_method.SetMetricMovingMask(moving_mask)

    # 设置B样条变换
    mesh_size = [max(3, int(sz/40)) for sz in fixed_image.GetSize()[:2]]
    transform = sitk.BSplineTransformInitializer(fixed_image, mesh_size, order=3)
    
    # 设置初始变换
    registration_method.SetInitialTransform(transform, True)

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
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # 重采样移动图像
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(final_transform)
    transformed_moving_image = resampler.Execute(moving_image)

    # 重采样移动mask
    if moving_mask is not None:
        mask_resampler = sitk.ResampleImageFilter()
        mask_resampler.SetReferenceImage(fixed_image)
        mask_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        mask_resampler.SetTransform(final_transform)
        transformed_moving_mask = mask_resampler.Execute(moving_mask)
        
        # 保存变换后的mask
        # output_mask_path = f"transformed_{os.path.basename(moving_mask.GetOrigin())}.nii.gz"
        # sitk.WriteImage(transformed_moving_mask, output_mask_path)
        # print(f"Transformed moving mask saved to: {output_mask_path}")

    return transformed_moving_image,transformed_moving_mask, final_transform


def main():
    base_dir="./output"
    fixed_image_path = 'registered_sub2092_nativet1mapt1_fixed_slice0.nii.gz'
    moving_image_paths = [
        'registered_sub2092_nativet1mapt1_rigid_slice0.nii.gz',
        'registered_sub2092_t2mapt2_rigid_slice0.nii.gz'
    ]
    moving_mask_paths = [
        'registered_sub2092_nativet1mapt1_rigid_mask_slice0.nii.gz',
        'registered_sub2092_t2mapt2_rigid_mask_slice0.nii.gz'
    ]

    fixed_image = sitk.ReadImage(os.path.join(base_dir,fixed_image_path))
    fixed_mask_path = 'registered_sub2092_nativet1mapt1_fixed_mask_slice0.nii.gz'
    fixed_mask = sitk.ReadImage(os.path.join(base_dir,fixed_mask_path))

    for moving_image_path, moving_mask_path in zip(moving_image_paths, moving_mask_paths):
        moving_image = sitk.ReadImage(os.path.join(base_dir,moving_image_path))
        moving_mask = sitk.ReadImage(os.path.join(base_dir,moving_mask_path))
        print(f"Registering {moving_image_path} to fixed image...")
        registered_image,registered_mask, final_transform = perform_bspline_registration(fixed_image, moving_image, fixed_mask=fixed_mask, moving_mask=moving_mask)

        # 添加调试信息
        if registered_image is not None:
            print(f"Registered Image Size: {registered_image.GetSize()}, Type: {registered_image.GetPixelIDTypeAsString()}")
        else:
            print(f"Error: Registered image is None for {moving_image_path}")

        output_path = f"registered_ffd_img_{moving_image_path}"

        sitk.WriteImage(registered_image, os.path.join(base_dir,output_path))
        
        output_path = f"registered_ffd_mask_{moving_image_path}"
        sitk.WriteImage(registered_mask, os.path.join(base_dir,output_path))
        print(f"Registered image saved to: {output_path}")


if __name__ == '__main__':
    main()
