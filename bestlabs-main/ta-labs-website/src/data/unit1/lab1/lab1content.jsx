import React from 'react';

function Lab1old() {
  return (
    <div className="Lab1">
      <section className="m-6 p-6 bg-white shadow-md rounded-lg">
        <h2 className="text-xl font-semibold text-[#113b7d]">Step 1: Install Python</h2>
        <ol className='open-sans-about-us-page list-decimal pl-6 pt-4 text-gray-700'>
          <li><strong>Visit the Python Official Website:</strong> Open your web browser and go to the official Python website. The URL is <a href="https://www.python.org" target="_blank" rel="noopener noreferrer" className="link-primary">python.org</a>.</li><br />
          <li><strong>Navigate to the Downloads Section:</strong> On the Python website, you'll find a navigation bar at the top. Hover your cursor over the "Downloads" tab, and you'll see a dropdown menu.</li><br />
          <li><strong>Choose Your Python Version:</strong> In the dropdown menu, you'll see different versions of Python available for download. Typically, you'll have two options: Python 3.x (the latest version) and Python 2.x (an older version, which is not recommended for new projects as it has reached its end-of-life). Click on the version you want to download. For most users, Python 3.x is the appropriate choice.</li><br />
          <li><strong>Select the Installer:</strong> Once you've selected the version, you'll be directed to the downloads page for that version. You'll see various installers available for different operating systems (Windows, macOS, Linux, etc.). Choose the installer that corresponds to your operating system. For example, if you're using Windows, select the Windows installer.</li><br />
          <li><strong>Download the Installer:</strong> Click on the download link for the installer, and your browser will start downloading the installer file. The file size may vary depending on your operating system and the version of Python you're downloading.</li><br />
          <li><strong>Run the Installer:</strong> Once the download is complete, locate the installer file on your computer (it's usually in your Downloads folder unless you specified a different location). Double-click the installer file to run it.</li><br />
          <li><strong>Install Python:</strong> The installer will launch a setup wizard. Follow the on-screen instructions to install Python on your computer. You can usually accept the default settings, but you may have the option to customize the installation (e.g., choosing the installation directory).</li><br />
          <li><strong>Check "Add Python to PATH" (Windows Only):</strong> On Windows, during the installation process, make sure to check the box that says "Add Python to PATH." This option ensures that Python is added to your system's PATH environment variable, allowing you to run Python from the command line more easily.</li><br />
          <li><strong>Complete the Installation:</strong> Once you've selected your preferences, click "Install" or "Finish" to complete the installation process. Python will be installed on your computer.</li><br />
          <li><strong>Verify the Installation:</strong> After the installation is complete, you can verify that Python has been installed correctly by opening a command prompt or terminal and typing <code>python --version</code> or <code>python3 --version</code> (depending on your operating system and configuration). This command should display the version of Python you've installed.</li><br />
        </ol>
      </section>

      <section className="m-6 p-6 bg-white shadow-md rounded-lg">
        <h2 className="text-xl font-semibold text-[#113b7d]">Step 2: Install Visual Studio Code</h2>
        <ol className='open-sans-about-us-page list-decimal pl-6 pt-4 text-gray-700'>
          <li><strong>Visit the Visual Studio Code Website:</strong> Open your web browser and go to the official Visual Studio Code website. The URL is <a href="https://code.visualstudio.com" target="_blank" rel="noopener noreferrer" className="link-primary">code.visualstudio.com</a>.</li><br />
          <li><strong>Download Visual Studio Code:</strong> On the homepage of the website, you'll see a prominent "Download" button. Click on it.</li><br />
          <li><strong>Select Your Operating System:</strong> Once you click the "Download" button, you'll be redirected to a page where you can choose the version of Visual Studio Code for your operating system. There are options available for Windows, macOS, and Linux. Click on the download link for the version that matches your operating system.</li><br />
          <li><strong>Download Begins:</strong> After selecting your operating system, the download should start automatically. If it doesn't, you might need to click on a specific link to initiate the download.</li><br />
          <li><strong>Locate the Installer:</strong> Once the download is complete, locate the installer file on your computer. By default, it's usually in your Downloads folder unless you specified a different location.</li><br />
          <li><strong>Run the Installer:</strong> Double-click on the installer file to run it. This will launch the setup wizard.</li><br />
          <li><strong>Install Visual Studio Code:</strong> Follow the instructions in the setup wizard to install Visual Studio Code on your computer. You can usually accept the default settings, but you may have the option to customize the installation (e.g., choosing the installation directory).</li><br />
          <li><strong>Launch Visual Studio Code:</strong> Once the installation is complete, you can launch Visual Studio Code by finding it in your list of installed applications or by searching for it in your computer's search bar.</li><br />
          <li><strong>Optional: Configure Settings:</strong> Upon launching Visual Studio Code for the first time, you might want to configure some settings according to your preferences. This includes choosing a color theme, configuring keyboard shortcuts, installing extensions, etc.</li><br />
          <li><strong>Start Coding:</strong> You're now ready to start coding! Visual Studio Code is a powerful code editor with features like syntax highlighting, code completion, debugging support, and more.</li><br />
        </ol>
      </section>

      <section className="m-6 p-6 bg-white shadow-md rounded-lg">
        <h2 className="text-xl font-semibold text-[#113b7d]">Step 3: Install Additional Python Libraries</h2>
        <ol className='open-sans-about-us-page list-decimal pl-6 pt-4 text-gray-700'>
          <li><strong>Open Command Prompt or Terminal:</strong> Depending on your operating system, open Command Prompt (Windows) or Terminal (macOS/Linux).</li><br />
          <li><strong>Check if pip is Installed:</strong> Type the following command and press Enter to check if pip, Python's package manager, is installed:</li><br />
          <code className="bg-gray-100 p-2 rounded-md">pip --version</code><br />
          <li><strong>If pip is installed:</strong> you'll see its version number. If it's not installed, you'll need to install Python first, as pip comes with Python installations starting from Python 3.4.</li><br />
          <li><strong>Upgrade pip (Optional):</strong> It's a good practice to upgrade pip to the latest version. You can do this by running the following command:</li><br />
          <code className="bg-gray-100 p-2 rounded-md">pip install --upgrade pip</code><br />
          <li><strong>Install numpy:</strong> Type the following command and press Enter to install the numpy library:</li><br />
          <code className="bg-gray-100 p-2 rounded-md">pip install numpy</code><br />
          <li><strong>Install scipy:</strong> Next, install the scipy library by running the following command:</li><br />
          <code className="bg-gray-100 p-2 rounded-md">pip install scipy</code><br />
          <li><strong>Install matplotlib:</strong> Install the matplotlib library for data visualization with the following command:</li><br />
          <code className="bg-gray-100 p-2 rounded-md">pip install matplotlib</code><br />
          <li><strong>Install scikit-learn:</strong> Install scikit-learn by running the following command:</li><br />
          <code className="bg-gray-100 p-2 rounded-md">pip install scikit-learn</code><br />
          <li><strong>Install pandas:</strong> Finally, install the pandas library for data manipulation and analysis with the following command:</li><br />
          <code className="bg-gray-100 p-2 rounded-md">pip install pandas</code><br />
        </ol>
      </section>

      <footer className="bg-[#f7f7f7] p-6">
        <p className="text-gray-700 text-lg">
          This lab experiment aims to familiarize students with the setup of an AI development environment, including the installation of necessary software and tools.
        </p>
      </footer>
    </div>
  );
}

export default Lab1old;
