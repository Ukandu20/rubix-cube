from cube import MOVE_TO_ACTION, CubeEnvironment

env = CubeEnvironment()

print(env.render())

env.step(MOVE_TO_ACTION["U"])
print(env.render())

env.scramble(1)
print(env.render())
