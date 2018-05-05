import time
import run
import matplotlib.pyplot as plt
import numpy as np

def plot_graph(birds=1, episode_number=5):
    bird_rewards = []
    random_rewards = []
    
    # Collect returns of all agents playing episodes
    for i in range(birds):
        bird_rewards.append(run.play(episodes=episode_number))
        random_rewards.append(run.random_play(episodes=episode_number))
        print("Bird", i)

    # Average across birds and 10 episodes - for both agents
    bird_rewards = np.mean(bird_rewards, axis=0, dtype=np.float64)
    bird_rewards = np.mean(bird_rewards.reshape(-1, 10), axis=1)
    random_rewards = np.mean(random_rewards, axis=0, dtype=np.float64)
    random_rewards = np.mean(random_rewards.reshape(-1, 10), axis=1)
    x_axis = range(10, (len(bird_rewards)+1)*10, 10)

    # Plot figure
    plt.figure(1)
    plt.plot(x_axis, bird_rewards)
    plt.plot(x_axis, random_rewards)
    plt.title("Average learning of " + str(birds) + " birds")
    plt.xlabel("Episode Number")
    plt.ylabel("Reward averaged over 10 episodes")
    plt.legend(["Agent", "Random"], loc = "best")
    plt.show()

    # Save and return
    np.save('bird_learning', bird_rewards)
    return bird_rewards

# Plot graph and calculate time
t = time.process_time()
plot_graph(birds=10, episode_number=350)
elapsed_time = round(((time.process_time()-t)/60),2)
print("Time taken:", elapsed_time, "minutes")
