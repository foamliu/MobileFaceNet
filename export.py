import time

import torch

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    print('loading {}...'.format(checkpoint))
    start = time.time()
    checkpoint = torch.load(checkpoint)
    print('elapsed {} sec'.format(time.time() - start))
    model = checkpoint['model'].module
    print(type(model))

    # model.eval()
    filename = 'mobilefacenet.pt'
    print('saving {}...'.format(filename))
    start = time.time()
    torch.save(model.state_dict(), filename)
    print('elapsed {} sec'.format(time.time() - start))

    scripted_model_file = 'mobilefacenet_scripted.pt'
    # Fuses modules
    model.fuse_model()
    torch.jit.save(torch.jit.script(model), scripted_model_file)
