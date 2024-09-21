import imageio
import os
import matplotlib.pyplot as plt


def visualize_depth(depth_map):
    depth_map = depth_map.squeeze().cpu().detach().numpy()
    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar(label="Depth")
    plt.title("Depth Map")
    plt.show()

def visualize_ego_motion(ego_motion):
    # ego_motion is expected to be a 6D vector
    translation = ego_motion[:, :3]
    rotation = ego_motion[:, 3:]
    
    print("Translation Vector (x, y, z):", translation.cpu().detach().numpy())
    print("Rotation Vector (Euler Angles):", rotation.cpu().detach().numpy())

def visualize_optical_flow(flow):
    flow = flow.squeeze().cpu().detach().numpy()  # Shape: (2, H, W)
    u = flow[0]  # Horizontal flow
    v = flow[1]  # Vertical flow

    # Visualize with quiver plot
    plt.figure(figsize=(10, 10))
    plt.quiver(u, v, angles="xy", scale_units="xy", scale=1, color="r")
    plt.title("Optical Flow (Quiver Plot)")
    plt.show()


def create_depth_gif(depth_sequence, output_file="depth_animation.gif"):
    # depth_sequence: A list of depth maps (each as a torch.Tensor or numpy array)
    
    images = []
    for i, depth_map in enumerate(depth_sequence):
        # Plot and save each frame as an image
        fig, ax = plt.subplots()
        depth_map = depth_map.squeeze().cpu().detach().numpy()
        ax.imshow(depth_map, cmap='plasma')
        ax.set_title(f"Depth Map Frame {i}")
        plt.colorbar(ax.imshow(depth_map, cmap='plasma'))
        
        # Save the frame
        frame_path = f"depth_frame_{i}.png"
        plt.savefig(frame_path)
        images.append(imageio.imread(frame_path))  # Append to image list for gif creation
        plt.close(fig)  # Close the plot
        
        # Remove the temporary image
        os.remove(frame_path)
    
    # Save the frames as a GIF
    imageio.mimsave(output_file, images, fps=5)  # fps: frames per second
    print(f"Depth GIF saved at {output_file}")

def create_ego_motion_gif(ego_motion_sequence, output_file="ego_motion_animation.gif"):
    images = []
    
    for i, ego_motion in enumerate(ego_motion_sequence):
        fig, ax = plt.subplots()
        
        translation = ego_motion[:, :3].cpu().detach().numpy()  # Extract translation
        rotation = ego_motion[:, 3:].cpu().detach().numpy()  # Extract rotation
        
        # Create a bar plot for translation and rotation
        ax.bar(["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"], list(translation[0]) + list(rotation[0]))
        ax.set_ylim([-2, 2])  # Set limit to keep scaling consistent
        
        ax.set_title(f"Ego-Motion Frame {i}")
        
        # Save the frame
        frame_path = f"ego_motion_frame_{i}.png"
        plt.savefig(frame_path)
        images.append(imageio.imread(frame_path))
        plt.close(fig)
        
        # Remove the temporary image
        os.remove(frame_path)
    
    # Save the frames as a GIF
    imageio.mimsave(output_file, images, fps=5)
    print(f"Ego-Motion GIF saved at {output_file}")

def create_optical_flow_gif(flow_sequence, output_file="flow_animation.gif"):
    images = []
    
    for i, flow in enumerate(flow_sequence):
        fig, ax = plt.subplots()
        
        # Extract the flow (flow should be of shape (batch, 2, H, W), for simplicity we take [0] as the batch)
        flow = flow[0].cpu().detach().numpy()  # (2, H, W)
        u = flow[0]  # Horizontal flow
        v = flow[1]  # Vertical flow
        
        ax.quiver(u, v, angles="xy", scale_units="xy", scale=1, color="r")
        ax.set_title(f"Optical Flow Frame {i}")
        
        # Save the frame
        frame_path = f"flow_frame_{i}.png"
        plt.savefig(frame_path)
        images.append(imageio.imread(frame_path))
        plt.close(fig)
        
        # Remove the temporary image
        os.remove(frame_path)
    
    # Save the frames as a GIF
    imageio.mimsave(output_file, images, fps=5)
    print(f"Optical Flow GIF saved at {output_file}")
