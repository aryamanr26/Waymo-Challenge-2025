# waymo_helpers.py

import uuid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

def create_figure_and_axes(size_pixels):
    fig, ax = plt.subplots(1, 1, num=uuid.uuid4())
    dpi = 100
    size_inches = size_pixels / dpi
    fig.set_size_inches([size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='y', colors='black')
    fig.set_tight_layout(True)
    ax.grid(False)
    return fig, ax

def fig_canvas_image(fig):
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.98)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

def get_colormap(num_agents):
    colors = cm.get_cmap('jet', num_agents)
    colors = colors(range(num_agents))
    np.random.shuffle(colors)
    return colors

def get_colormap_by_type(agent_types, sdc_index):
    colors = np.zeros((len(agent_types), 4))
    for i, t in enumerate(agent_types):
        if t == 1 and i != sdc_index:
            colors[i] = [0, 0, 1, 1]
        elif t == 2:
            colors[i] = [1, 0, 0, 1]
        elif i == sdc_index:
            colors[i] = [0, 0, 0, 1]
        else:
            colors[i] = [0, 1, 0, 1]
    return colors

def get_viewport(all_states, all_states_mask):
    valid_states = all_states[all_states_mask]
    all_y = valid_states[..., 1]
    all_x = valid_states[..., 0]
    center_y = (np.max(all_y) + np.min(all_y)) / 2
    center_x = (np.max(all_x) + np.min(all_x)) / 2
    width = max(np.ptp(all_y), np.ptp(all_x))
    return center_y, center_x, width

def visualize_one_step(states, mask, roadgraph, title, center_y, center_x, width, color_map, size_pixels=1000):
    fig, ax = create_figure_and_axes(size_pixels=size_pixels)
    rg_pts = roadgraph[:, :2].T
    ax.plot(rg_pts[0, :], rg_pts[1, :], 'k.', alpha=1, ms=2)
    masked_x = states[:, 0][mask]
    masked_y = states[:, 1][mask]
    colors = color_map[mask]
    ax.scatter(masked_x, masked_y, marker='o', linewidths=3, color=colors)
    ax.set_title(title)
    size = max(10, width * 1.0)
    ax.axis([center_x - size/2, center_x + size/2, center_y - size/2, center_y + size/2])
    ax.set_aspect('equal')
    image = fig_canvas_image(fig)
    plt.close(fig)
    return image

def state_space(data):
    past_states = tf.stack([data['state/past/x'], data['state/past/y'], data['state/past/z']], -1).numpy()

def action_space(data):
    pass

def visualize_all_agents_smooth(decoded_example, size_pixels=1000):
    past_states = tf.stack([decoded_example['state/past/x'], decoded_example['state/past/y']], -1).numpy()
    past_states_mask = decoded_example['state/past/valid'].numpy() > 0.0
    current_states = tf.stack([decoded_example['state/current/x'], decoded_example['state/current/y']], -1).numpy()
    current_states_mask = decoded_example['state/current/valid'].numpy() > 0.0
    future_states = tf.stack([decoded_example['state/future/x'], decoded_example['state/future/y']], -1).numpy()
    future_states_mask = decoded_example['state/future/valid'].numpy() > 0.0
    roadgraph_xyz = decoded_example['roadgraph_samples/xyz'].numpy()
    object_type = decoded_example['state/type'].numpy()
    is_sdc = decoded_example['state/is_sdc'].numpy()
    sdc_index = np.where(is_sdc == 1)[0]
    color_map = get_colormap_by_type(object_type, sdc_index[0] if len(sdc_index) > 0 else -1)

    all_states = np.concatenate([past_states, current_states, future_states], 1)
    all_states_mask = np.concatenate([past_states_mask, current_states_mask, future_states_mask], 1)
    center_y, center_x, width = get_viewport(all_states, all_states_mask)

    images = []
    for i, (s, m) in enumerate(zip(np.split(past_states, past_states.shape[1], 1),
                                   np.split(past_states_mask, past_states_mask.shape[1], 1))):
        images.append(visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz, f'past: {past_states.shape[1] - i}', center_y, center_x, width, color_map, size_pixels))

    images.append(visualize_one_step(current_states[:, 0], current_states_mask[:, 0], roadgraph_xyz, 'current', center_y, center_x, width, color_map, size_pixels))

    for i, (s, m) in enumerate(zip(np.split(future_states, future_states.shape[1], 1),
                                   np.split(future_states_mask, future_states_mask.shape[1], 1))):
        images.append(visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz, f'future: {i + 1}', center_y, center_x, width, color_map, size_pixels))

    return images

num_map_samples = 30000

# Example field definition
roadgraph_features = {
    'roadgraph_samples/dir': tf.io.FixedLenFeature(
        [num_map_samples, 3], tf.float32, default_value=None
    ),
    'roadgraph_samples/id': tf.io.FixedLenFeature(
        [num_map_samples, 1], tf.int64, default_value=None
    ),
    'roadgraph_samples/type': tf.io.FixedLenFeature(
        [num_map_samples, 1], tf.int64, default_value=None
    ),
    'roadgraph_samples/valid': tf.io.FixedLenFeature(
        [num_map_samples, 1], tf.int64, default_value=None
    ),
    'roadgraph_samples/xyz': tf.io.FixedLenFeature(
        [num_map_samples, 3], tf.float32, default_value=None
    ),
}
# Features of other agents.
state_features = {
    'state/id':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/is_sdc':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/current/bbox_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/timestamp_micros':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/vel_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/future/bbox_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/timestamp_micros':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/vel_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/past/bbox_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/timestamp_micros':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/vel_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
}

traffic_light_features = {
    'traffic_light_state/current/state':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/valid':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/x':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/y':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/z':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/past/state':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/valid':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/x':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/y':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/z':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
}

features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)