# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: download_png.py
# @Last modified by:   Joey Teng
# @Last modified time: 03-May-2018
import time

from selenium import webdriver
from selenium.webdriver.firefox.options import Options

FIREFOX_BINARY = ''.join([
    '/Applications/Firefox Developer Edition.app/Contents/MacOS/firefox-bin'])
WAIT_TIME = 3
DOWNLOAD_TIME = 2


def download(path, url):
    print("Initialising the Downloader using headless Firefox", flush=True)
    options = Options()
    options.set_headless()

    profile = webdriver.FirefoxProfile()
    profile.set_preference('browser.download.folderList', 2)
    profile.set_preference(
        'browser.download.manager.showWhenStarting',
        False)
    profile.set_preference('browser.download.dir', path)
    profile.set_preference(
        'browser.helperApps.neverAsk.saveToDisk',
        'image/png')

    profile.set_preference(
        "browser.preferences.defaultPerformanceSettings.enabled", False)
    profile.set_preference(
        "browser.shell.didSkipDefaultBrowserCheckOnFirstRun", False)
    profile.set_preference("browser.shell.checkDefaultBrowser", False)
    profile.set_preference("dom.ipc.processCount", 0)
    profile.set_preference("dom.ipc.plugins.enabled", False)
    profile.set_preference("browser.tabs.remote.autostart", False)

    driver = webdriver.Firefox(
        firefox_profile=profile,
        firefox_binary=FIREFOX_BINARY,
        options=options,
        log_path='/dev/null')

    print("Headless Firefox Downloader Initialised. Path: {}".format(
        path), flush=True)

    print("Loading {}...".format(url), flush=True)
    try:
        driver.get(url)
    except ConnectionRefusedError:
        print("Connection Refused: {}\n Skip...".format(url), flush=True)
        raise RuntimeError("Connection Refused")

    print("Page Loaded. Downloading {}...".format(url), flush=True)
    time.sleep(DOWNLOAD_TIME)
    driver.quit()
    print("Downloaded {}".format(url), flush=True)


if __name__ == '__main__':
    pass
