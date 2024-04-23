import gc
import torch
import torchvision

from .utils import concat_along_channel_dim

# defining class for video dataset
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, mp4_file_list, augmentations = None):
        super(VideoDataset, self).__init__()
        # list of paths to mp4 files
        self.mp4_file_list = mp4_file_list
        # (callable) optional albumentations tranforms 
        self.augmentations = augmentations
    
    def __len__(self):
        # getting number of videos in the underlying list
        num_videos = len(self.mp4_file_list)
        return num_videos
    
    def __getitem__(self, idx):
        # get file name
        mp4_file = self.mp4_file_list[idx]
        # read video
        video, _, _ = torchvision.io.read_video(mp4_file, pts_unit = 'sec') # shape: (num_frames, height, width, num_channels)
        # normalize pixels in [0, 1]
        video = video / 255 
        # apply optional augmentations
        # need to apply the augmentations 
        # for each frame of the video 
        if self.augmentations:
            # initializing the list for storing
            # each transformed frame
            transformed_frames_store = []
            # iterating over each frame of the vide
            for frame in video:
                # converting frame tensor to np array
                # as required by albumentations
                frame = frame.numpy()
                # applying augmentations on array
                frame = self.augmentations(image = frame)["image"]
                # converting array back to torch tensor
                frame = torch.from_numpy(frame)
                # appending transformed frame to the store
                transformed_frames_store.append(frame)
            # concatenating back the frames over the time dimension
            video = torch.stack(transformed_frames_store, dim = 0)
        # converting back to shape (num_frames, num_channels, height, width)
        video = torch.permute(video, (0, 3, 2, 1))
        return video

# defining the VQGAN latent video dataset
class VQGANLatentVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, encoder, device):
        super(VQGANLatentVideoDataset, self).__init__()
        self.base_dataset = base_dataset # instance of VideoDataset class
        self.encoder = encoder # instance of river.model.encoder.Encoder class
        self.device = device # device on which the encoder is run

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # getting video and moving it to device
        video = self.base_dataset[idx].to(self.device)
        # encoding latent video, detaching it from computation graph 
        # as we don't need the gradients wrt the encoder and moving
        # it to cpu for saving up gpu memory
        latent_video = self.encoder(video).detach().cpu()
        # deleting unused video and cleaning up memory and gpu cache
        del video
        gc.collect()
        torch.cuda.empty_cache()
        return latent_video

# class for handling the paired lagged datasets
class LaggedDataset(torch.utils.data.Dataset):
    def __init__(self, current_states, next_states, device):
        super(LaggedDataset, self).__init__()
        self.current_states = current_states
        self.next_states = next_states
        self.device = device
    
    def __len__(self):
        return self.next_states.shape[0]
    
    def __getitem__(self, idx):
        current_state = self.current_states[idx].to(self.device)
        next_state = self.next_states[idx].to(self.device)
        batch_dict = {
            "current_state": current_state, 
            "next_state": next_state,
            "conditioning_state": current_state
        }
        return batch_dict

# class for handling the paired lagged datasets with random context
class LaggedDatasetWithRandomContext(torch.utils.data.Dataset):
    def __init__(self, current_states, next_states, random_context_states, device):
        super(LaggedDatasetWithRandomContext, self).__init__()
        self.current_states = current_states
        self.next_states = next_states
        self.random_context_states = random_context_states
        self.device = device
            
    def __len__(self):
        return self.next_states.shape[0]
    
    def __getitem__(self, idx):
        current_state = self.current_states[idx].to(self.device) # shape: num_channels, height, width
        next_state = self.next_states[idx].to(self.device) # shape: num_channels, height, width
        random_context_state = self.random_context_states[idx].to(self.device) # shape: num_channels, height, width
        conditioning_state = concat_along_channel_dim(current_state, random_context_state)
        batch_dict = {
            "current_state": current_state, 
            "next_state": next_state, 
            "conditioning_state": conditioning_state, 
        }
        return batch_dict