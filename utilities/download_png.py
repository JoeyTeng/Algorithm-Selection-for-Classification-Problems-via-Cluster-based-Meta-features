# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: download_png.py
# @Last modified by:   Joey Teng
# @Last modified time: 03-May-2018
import time
import weakref

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.keys import Keys

FIREFOX_BINARY = ''.join([
    '/Applications/Firefox Developer Edition.app/Contents/MacOS/firefox-bin'])
WAIT_TIME = 10
DOWNLOAD_TIME = 2


def to_del(downloader):
    if downloader is not None and not downloader.clean:
        print("Preparing to destroy the driver in {}s...".format(
            WAIT_TIME), flush=True)
        time.sleep(WAIT_TIME)
        downloader.driver.quit()
        downloader.clean = True


class Downloader(object):
    def __init__(self):
        self.initialised = False
        self.clean = True

    def initialise(self, path):
        print("Initialising the Downloader using headless Firefox", flush=True)
        options = Options()
        options.add_argument("--headless")

        profile = webdriver.FirefoxProfile()
        profile.set_preference('browser.download.folderList', 2)
        profile.set_preference(
            'browser.download.manager.showWhenStarting',
            False)
        profile.set_preference('browser.download.dir', path)
        profile.set_preference(
            'browser.helperApps.neverAsk.saveToDisk',
            'image/png')

        self.driver = webdriver.Firefox(
            firefox_profile=profile,
            firefox_binary=FIREFOX_BINARY,
            options=options)

        weakref.finalize(self, to_del, self)

        self.initialised = True
        self.clean = False
        print("Headless Firefox Downloader Initialised. Path: {}".format(
            path), flush=True)

    def download(self, url):
        print("Loading {}...".format(url), flush=True)
        try:
            self.driver.get(url)
        except ConnectionRefusedError:
            print("Connection Refused: {}\n Skip...".format(url), flush=True)
            raise RuntimeError("Connection Refused")

        print("Page Loaded. Downloading {}...".format(url), flush=True)
        time.sleep(DOWNLOAD_TIME)
        # self.driver.find_element_by_tag_name(
        #     'body').send_keys(Keys.COMMAND + 'w')
        # self.driver.switch_to.window(self.driver.window_handles[0])
        # self.driver.close()
        print("Downloaded {}".format(url), flush=True)

    def on_del(self):
        if not self.clean:
            print("Preparing to destroy the driver in {}s...".format(
                WAIT_TIME), flush=True)
            time.sleep(WAIT_TIME)
            self.driver.quit()
            self.clean = True


if __name__ == '__main__':
    pass
