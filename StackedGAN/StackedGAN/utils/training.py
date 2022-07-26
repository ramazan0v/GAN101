import math
import numpy as np
from utils.image import to_dirname, check_dir
from PIL import Image

def display(step, steps, D_loss, GAN_loss):
    """
        step : Integer
        steps : Integer
        D_loss : Float
        GAN_loss : Float
    """
    #print('Step: '+str(step+1)+'/'+str(steps)+' - D loss: '+str(D_loss)+' - GAN loss: '+str(GAN_loss), end='\r\n')
    #print('-------------------')
    #print('\n')


def train(G, D, GAN, sets, batch, d1_losses, cum_d1_loss, gan1_losses, cum_gan1_loss):
    """
        G : Keras model
        D : Keras model
        GAN : Keras model, GAN
        sets : 
        batch : Integer
        loss : Tuple
    """
    np.random.shuffle(sets)
    input_dim = G.input_shape[1]
    steps = math.ceil(len(sets) / batch)
    for step in range(steps):
        real = sets[step*batch:(step+1)*batch]
        samples = len(real)

        answer = np.ones(samples)
        d1_loss_real = D.train_on_batch(x=real, y=answer)

        noise = np.random.uniform(0, 1, size=(samples, input_dim))
        generated = G.predict(noise)
        answer = np.zeros(samples)
        d1_loss_fake = D.train_on_batch(x=generated, y=answer)

        noise = np.random.uniform(0, 1, size=(samples, input_dim))
        answer = np.ones(samples)
        GAN_loss = GAN.train_on_batch(x=noise, y=answer)
        
        
        
        d1_loss_this_step = (0.5 * np.add(d1_loss_real, d1_loss_fake))
        cum_d1_loss += d1_loss_this_step
        d1_losses.append(d1_loss_this_step)
        
        cum_gan1_loss += GAN_loss
        gan1_losses.append(GAN_loss)

        
        
    display(step, steps, cum_d1_loss, cum_gan1_loss)
    #print()
    return (d1_losses, cum_d1_loss, gan1_losses, cum_gan1_loss)


def train_with_images(G_before, G, D, GAN, sets, batch, d2_losses, cum_d2_loss, gan2_losses, cum_gan2_loss):
    """
        G_before : Keras model
        G : Keras model
        D : Keras model
        GAN : Keras model, GAN
        sets :
        batch : Integer
        loss : Tuple
    """
    np.random.shuffle(sets)
    input_dim = G_before.input_shape[1]
    steps = math.ceil(len(sets) / batch)
    for step in range(steps):
        real = sets[step*batch:(step+1)*batch]
        samples = len(real)

        answer = np.ones(samples)
        d2_loss_real = D.train_on_batch(x=real, y=answer)

        noise = np.random.uniform(0, 1, size=(samples, input_dim))
        G_out = G_before.predict(noise)
        generated = G.predict(G_out)
        answer = np.zeros(samples)
        d2_loss_fake = D.train_on_batch(x=generated, y=answer)

        noise = np.random.uniform(0, 1, size=(samples, input_dim))
        G_out = G_before.predict(noise)
        answer = np.ones(samples)
        GAN_loss = GAN.train_on_batch(x=G_out, y=answer)
        
        
        
        d2_loss_this_step = (0.5 * np.add(d2_loss_real, d2_loss_fake))
        cum_d2_loss += d2_loss_this_step
        d2_losses.append(d2_loss_this_step)
        
        cum_gan2_loss += GAN_loss
        gan2_losses.append(GAN_loss)

        
        

    display(step, steps, cum_d2_loss, cum_gan2_loss)
    #print()
    return (d2_losses, cum_d2_loss, gan2_losses, cum_gan2_loss)

def train_with_images4x(G1, G2, G, D, GAN, sets, batch, d3_losses, cum_d3_loss, gan3_losses, cum_gan3_loss):
    np.random.shuffle(sets)
    input_dim = G1.input_shape[1]
    steps = math.ceil(len(sets) / batch)
    for step in range(steps):
        real = sets[step*batch:(step+1)*batch]
        samples = len(real)

        answer = np.ones(samples)
        d3_loss_real = D.train_on_batch(x=real, y=answer)

        noise = np.random.uniform(0, 1, size=(samples, input_dim))
        G_out1 = G1.predict(noise)
        G_out2 = G2.predict(G_out1)
        generated = G.predict(G_out2)
        answer = np.zeros(samples)
        d3_loss_fake = D.train_on_batch(x=generated, y=answer)

        noise = np.random.uniform(0, 1, size=(samples, input_dim))
        G_out1 = G1.predict(noise)
        G_out2 = G2.predict(G_out1)
        answer = np.ones(samples)
        GAN_loss = GAN.train_on_batch(x=G_out2, y=answer)
        
        
        
        d3_loss_this_step = (0.5 * np.add(d3_loss_real, d3_loss_fake))
        cum_d3_loss += d3_loss_this_step
        d3_losses.append(d3_loss_this_step)
        
        cum_gan3_loss += GAN_loss
        gan3_losses.append(GAN_loss)

    display(step, steps, cum_d3_loss, cum_gan3_loss)
    #print()
    return (d3_losses, cum_d3_loss, gan3_losses, cum_gan3_loss)







def save_training_images(images, epoch=1, stack_no=1, batch=1, step=1):
    output_dirname = to_dirname('.\\ImagesUsed')
    check_dir(output_dirname)
    images = images * 255
    images = images.astype(np.uint8)
    for i in range(len(images)):
        image = Image.fromarray(images[i])
        image.save(output_dirname
                   + '\\epoch'+str(epoch)
                   + '_Stack_'+str(stack_no) 
                   + '_Batch_'+str(batch)
                   + '_Step_'+str(step)
                   + '_Image_'+str(i)+".bmp")
        

def train_stacks(epoch, number_of_stacks, stack_no, Gs, D, GAN, sets, batch, d_losses, cum_d_loss, gan_losses, cum_gan_loss):
    #print('train_stacks-number_of_stacks='+str(number_of_stacks)+' stack_no='+str(stack_no))
    ##np.random.shuffle(sets)
    input_dim = Gs[0].input_shape[1]
    steps = math.ceil(len(sets) / batch)
    for step in range(steps):

        real = sets[step*batch:(step+1)*batch]
        samples = len(real)
        
        #save_training_images(images=real, epoch=epoch, stack_no=stack_no, batch=batch, step=step)

        answer = np.ones(samples) ###################### Real=1
        d_loss_real = D.train_on_batch(x=real, y=answer)


        
        
        noise = np.random.uniform(0, 1, size=(samples, input_dim))
        G_input=[]
        G_input.append(noise)
        for i in range(stack_no + 1):
            G_out = Gs[i].predict(G_input[i])
            G_input.append(G_out)
        answer = np.zeros(samples) ###################### Fake=0
        d_loss_fake = D.train_on_batch(x=G_out, y=answer)

        
        
        answer = np.ones(samples) ###################### Real=1
        GAN_loss = GAN.train_on_batch(x=G_input[stack_no], y=answer)
        
        
        
        
        d_loss_this_step = (0.5 * np.add(d_loss_real, d_loss_fake))
        cum_d_loss += d_loss_this_step
        d_losses.append(d_loss_this_step)
        
        cum_gan_loss += GAN_loss
        gan_losses.append(GAN_loss)

    display(step, steps, cum_d_loss, cum_gan_loss)
    #print()
    return (d_losses, cum_d_loss, gan_losses, cum_gan_loss)