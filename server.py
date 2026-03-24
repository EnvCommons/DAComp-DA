from openreward.environments import Server

from dacomp_da import DACompDA

if __name__ == "__main__":
    server = Server([DACompDA])
    server.run()
