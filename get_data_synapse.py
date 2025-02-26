import synapseclient
from dotenv import load_dotenv


def main():
    load_dotenv()
    syn = synapseclient.login()
    dl_list_file_entities = syn.get_download_list()
    print(dl_list_file_entities)


if __name__ == "__main__":
    main()
