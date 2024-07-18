import jax
import flax.linen as nn
import flax
import gym
import jax.numpy as jnp


class GenericPolicy(nn.Module):
    action_dim: int
    state_dim: int
    hidden_dim: int = 64
    n_actions: int = 4

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        return x


class ValueFunction(nn.Module):
    state_dim: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


def policy_loss(policy, params, logits, states, actions, advantages, eps=0.2):
    current_logits = nn.logsoftmax(policy.apply(params, states))
    is_term = (jnp.exp(current_logits[actions]) - logits) * advantages
    positive_clip = (1 + eps) * advantages
    negative_clip = (1 - eps) * advantages

    return -jnp.mean(jnp.minimum(is_term, positive_clip, negative_clip))


def value_function_loss(value_fn, params, state, reward):
    advantages = reward - value_fn.apply(params, state)
    return jnp.mean(advantages**2), advantages


def train_policy(env, train_steps=100, lr=1e-3, max_ep_len=1000, n_rollouts=10, df=0.9):
    # initialize the model and optimizer
    policy = GenericPolicy(env.observation_space.shape[0], env.action_space.n)
    value_fn = ValueFunction(env.observation_space.shape[0])
    policy_optimizer = flax.optim.Adam(learning_rate=lr).create(policy)
    value_optimizer = flax.optim.Adam(learning_rate=lr).create(value_fn)
    policy_params = policy.init(
        jax.random.PRNGKey(0), jax.numpy.zeros((1, env.observation_space.shape[0]))
    )

    value_params = value_fn.init(
        jax.random.PRNGKey(0), jax.numpy.zeros((1, env.observation_space.shape[0]))
    )

    value_grad_fn = jax.value_and_grad(value_function_loss, has_aux=True)
    policy_grad_fn = jax.value_and_grad(policy_loss)
    # create gym environment
    env = gym.make(env)
    for step in range(train_steps):
        # collect data
        data = []
        total_reward = 0
        for _ in range(n_rollouts):
            ep = []
            state = env.reset()
            for _ in range(max_ep_len):
                logits = nn.logsoftmax(policy.apply(policy_params, state))
                action = jax.random.categorical(jax.random.PRNGKey(0), logits)
                action_prob = logits[action]
                next_state, reward, done, _ = env.step(action)
                # convert above line to dict
                total_reward += reward
                ep.append(
                    {
                        "state": state,
                        "action": action,
                        "action_prob": action_prob,
                        "reward": reward,
                        "next_state": next_state,
                        "done": done,
                    }
                )
                if done:
                    break
                state = next_state
            # use discounting to attribute rewards for episodes
            for tm1, t in zip(ep[:-1][:-1:-1], ep[1:][::-1]):
                tm1["reward"] += df * t["reward"]
            data.extend(ep)
        print(f"Step: {step}, Average Episode Reward: {total_reward/n_rollouts}")
        # batch, compute advantage, and update value function
        states = jax.numpy.array([d["state"] for d in data])
        rewards = jax.numpy.array([d["reward"] for d in data])
        actions = jax.numpy.array([d["action"] for d in data])
        logits = jax.numpy.array([d["action_prob"] for d in data])
        # call jax value and grad to get advantadges and gradients+loss
        (value_loss, advantadges), grad = value_grad_fn(
            value_fn, value_params, states, rewards
        )

        # apply gradients to value function
        value_optimizer = value_optimizer.apply_gradient(grad)
        value_params = value_optimizer.target

        # update policy
        policy_loss, grad = policy_grad_fn(
            policy, policy_params, logits, states, actions, advantadges
        )
        policy_optimizer = policy_optimizer.apply_gradient(grad)
        policy_params = policy_optimizer.target
        print(f"Step: {step}, Policy Loss: {policy_loss}, Value Loss: {value_loss}")

    return policy_params, value_params


policy_params = train_policy(
    "CartPole-v1", train_steps=100, lr=1e-3, max_ep_len=100, n_rollouts=10, df=0.9
)
