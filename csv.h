#pragma once

typedef uint32_t uint32;

struct CSV
{
	struct Column
	{
		std::string			label;
		std::vector<float>	data;
	};

	std::vector<Column> columns;

	void SetColumnLabel(uint32 column, const char* label)
	{
		if (columns.size() <= column)
			columns.resize(column + 1);
		columns[column].label = label;
	}

	void SetData(uint32 column, uint32 row, float data)
	{
		if (columns.size() <= column)
			columns.resize(column + 1);
		if (columns[column].data.size() <= row)
			columns[column].data.resize(row + 1);
		columns[column].data[row] = data;
	}

	float GetData(uint32 column, uint32 row) const
	{
		if (columns.size() <= column)
			return 0.0f;
		if (columns[column].data.size() <= row)
			return 0.0f;
		return columns[column].data[row];
	}

	void SetDataRunningAverage(uint32 column, uint32 row, float newData, uint32 sampleIndex)
	{
		float data = GetData(column, row);
		data = Lerp(data, newData, 1.0f / float(sampleIndex + 1));
		SetData(column, row, data);
	}

	bool Save(const char* fileNameFormat, ...) const
	{
		// make filename
		char fileName[1024];
		va_list args;
		va_start(args, fileNameFormat);
		vsprintf_s(fileName, fileNameFormat, args);
		va_end(args);

		// open file
		FILE* file;
		fopen_s(&file, fileName, "wb");
		if (!file)
			return false;

		// header row
		fprintf(file, "row");
		for (uint32 column = 0; column < columns.size(); ++column)
			fprintf(file, ",\"%s\"", columns[column].label.c_str());
		fprintf(file, "\n");

		// data rows
		uint32 row = 0;
		bool moreRows = true;
		while (moreRows)
		{
			moreRows = false;
			for (uint32 column = 0; column < columns.size(); ++column)
			{
				const Column& col = columns[column];
				if (row < col.data.size())
				{
					moreRows = true;
					break;
				}
			}
			if (!moreRows)
				break;

			fprintf(file, "\"%u\"", row);
			for (uint32 column = 0; column < columns.size(); ++column)
			{
				const Column& col = columns[column];
				if (row < col.data.size())
					fprintf(file, ",\"%f\"", col.data[row]);
				else
					fprintf(file, ",\"%f\"", 0.0f);
			}
			fprintf(file, "\n");
			row++;
		}

		// close file
		fclose(file);
		return true;
	}

private:
	static float Lerp(float A, float B, float t)
	{
		return A * (1.0f - t) + B * t;
	}
};
