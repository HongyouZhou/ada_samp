import os
import numpy as np
import OpenEXR
import Imath
import array
import colorcet


def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices]  # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.0
    scale = np.amax(maxs - mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result


xml_head = """
<scene version="3.0.0">
    <integrator type="path">
        <integer name="max_depth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="far_clip" value="100"/>
        <float name="near_clip" value="0.1"/>
        <transform name="to_world">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        
        <sampler type="independent">
            <integer name="sample_count" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1600"/>
            <integer name="height" value="1200"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="int_ior" value="1.46"/>
        <rgb name="diffuse_reflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""

xml_ball_segment = """
    <shape type="sphere">
        <float name="radius" value="0.02"/>
        <transform name="to_world">
            <translate x="{}" y="{}" z="{}"/>
            <scale value="0.7"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_tail = """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="to_world">
            <scale x="10" y="10" z="10"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="to_world">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""


def colormap(x, y, z):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


def get_seg_color(seg_idx):
    # Use glasbey palette for better color distinction
    palette = colorcet.glasbey
    # Convert seg_idx to a valid index for the palette
    idx = int(seg_idx) % len(palette)
    # Convert hex color to RGB
    hex_color = palette[idx]
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return [r, g, b]


def read_exr(exr_path):
    """Read an EXR file and return the RGB image as a numpy array."""
    exr_file = OpenEXR.InputFile(exr_path)

    # Get the data window
    dw = exr_file.header()["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Read the RGB channels
    channels = ["R", "G", "B"]
    rgb = []
    for channel in channels:
        # Read the channel
        channel_data = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
        # Convert to numpy array
        channel_array = np.frombuffer(channel_data, dtype=np.float32)
        channel_array = channel_array.reshape(height, width)
        rgb.append(channel_array)

    # Stack the channels
    rgb_image = np.stack(rgb, axis=-1)
    return rgb_image


def mitsuba(pcl, path, seg=None):
    xml_segments = [xml_head]

    # pcl = standardize_bbox(pcl, 2048)
    pcl = pcl[:, [2, 0, 1]]
    pcl[:, 0] *= -1
    h = np.min(pcl[:, 2])

    for i in range(pcl.shape[0]):
        if seg is None:
            color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5)
        else:
            # Use the new color mapping function for segmentation
            color = get_seg_color(seg[i])
        if h < -0.25:
            xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2] - h - 0.6875, *color))
        else:
            xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
    xml_segments.append(xml_tail)

    xml_content = str.join("", xml_segments)

    # Save XML file
    xml_path = path
    with open(xml_path, "w") as f:
        f.write(xml_content)

    # Generate EXR file path
    exr_path = xml_path.replace(".xml", ".exr")

    # Run mitsuba command to render
    os.system(f"mitsuba {xml_path}")

    # Read and return the EXR file
    if os.path.exists(exr_path):
        return read_exr(exr_path)
    return None


if __name__ == "__main__":
    item = 10
    split = "train"
    dataset_name = "shapenetpart"
    root = "/ssdArray/hongyou/dev/data/shapenet"
    save_root = os.path.join("image", dataset_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    from dataset import ShapeNetPart

    d = ShapeNetPart(root=root, dataset_name=dataset_name, num_points=2048, split=split, random_rotate=False, load_name=True, segmentation=True)
    print("datasize:", d.__len__())

    pts, lb, seg, n, _ = d[item]
    print(pts.size(), pts.type(), lb.size(), lb.type(), seg.size(), seg.type(), n)
    xml_path = os.path.join(save_root, dataset_name + "_" + split + str(item) + "_" + str(n) + ".xml")

    # Get the rendered image
    image = mitsuba(pts.numpy(), xml_path, seg.numpy())

    if image is not None:
        # Save as PNG for visualization
        png_path = xml_path.replace(".xml", ".png")
        # Convert to 8-bit and save
        image_8bit = np.clip(image * 255, 0, 255).astype(np.uint8)
        from PIL import Image

        Image.fromarray(image_8bit).save(png_path)
        print(f"Saved rendered image to {png_path}")
