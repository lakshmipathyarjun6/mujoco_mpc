#include "JSONUtils.hpp"

#include "filereadstream.h"
#include "filewritestream.h"
#include "prettywriter.h"
#include "stringbuffer.h"

Document loadJSON(string inputFile)
{
    Document d;
    FILE *fp = fopen(inputFile.c_str(), "r"); // non-Windows use "r"

    cout << "Loading JSON file " << inputFile << endl;

    char readBuffer[65536];
    FileReadStream is(fp, readBuffer, sizeof(readBuffer));

    d.ParseStream(is);

    fclose(fp);

    return d;
}

void writeJSON(Document &d, string outputFile)
{
    FILE *fp = fopen(outputFile.c_str(), "w"); // non-Windows use "w"

    if (fp != NULL)
    {
        cout << "Writing JSON file " << outputFile << endl;

        char writeBuffer[65536];
        FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));

        PrettyWriter<FileWriteStream> writer(os);
        d.Accept(writer);

        fclose(fp);
    }
    else
    {
        cout << "Failed to get file pointer" << endl;
    }
}
