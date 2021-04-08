#ifndef CANVAS
#define CANVAS

#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <cmath>
using namespace std;

class Pixel
{
public:
    int x = 0, y = 0, R = 0, G = 0, B = 0;
    Pixel() {}
    Pixel(int x, int y, int R, int G, int B)
    {
        this->x = x;
        this->y = y;
        this->R = R;
        this->G = G;
        this->B = B;
    }
    friend class BMP_Image;
    int operator ==(const Pixel& arg)
    {
        if (R == arg.R && G == arg.G && B == arg.B)
        {
            return 1;
        }
        return 0;
    }
    int operator !=(const Pixel& arg)
    {
        if (R != arg.R || G != arg.G || B != arg.B)
        {
            return 1;
        }
        return 0;
    }
    Pixel& operator=(const Pixel& arg)
    {
        x = arg.x;
        y = arg.y;
        R = arg.R;
        G = arg.G;
        B = arg.B;
        return *this;
    }
};


class BMP_Image
{
private:
    // File Header
    uint16_t signature = 0x4D42; // BM file signature
    uint32_t file_size; // Size of file
    uint16_t reserved_1 = 0; // Reserved field 1
    uint16_t reserved_2 = 0; // Reserved field 2
    uint32_t bytes_shift = 54; // Shift of color bytes
    // Info Header
    uint32_t header_size = 40; // 40?
    //int32_t width;
    //int32_t height;
    uint16_t planes = 1; // number of planes
    uint16_t bits_per_pixel = 24; // Bits required for one pixel
    uint32_t compression = 0; // No compression
    uint32_t image_size = 0;
    uint32_t Xpix_per_m = 0; // Horizontal rezolution
    uint32_t Ypix_per_m = 0; // Vertical resolution
    uint32_t colors_used = 0; // Number of used colors
    uint32_t colors_important = 0; // Number of Important colors
    // Pixel data
    //uint8_t* pixels; // Pixels by bytes
    //int pixlen = 0;
public:
    // Modified block only for CUDA !
    uint8_t* pixels;
    int32_t width;
    int32_t height;
    int pixlen = 0;
    /* ************************* */
    BMP_Image()
    {
        width = 0;
        height = 0;
        file_size = 0;
        pixels = nullptr;
    }
    BMP_Image(string filename)
    {
        int load_status = load(filename);
        if (!load_status)
        {
            cout << "Error occured while tried to load image. " << endl;
        }
    }
    ~BMP_Image()
    {
        delete[] pixels;
    }
    BMP_Image(const BMP_Image& arg)
    {
        this->bits_per_pixel = arg.bits_per_pixel;
        this->bytes_shift = arg.bytes_shift;
        this->colors_important = arg.colors_important;
        this->colors_used = arg.colors_used;
        this->compression = arg.compression;
        this->file_size = arg.file_size;
        this->header_size = arg.header_size;
        this->height = arg.height;
        this->width = arg.width;
        this->image_size = arg.image_size;
        this->signature = arg.signature;
        this->reserved_1 = arg.reserved_1;
        this->reserved_2 = arg.reserved_2;
        this->planes = arg.planes;
        this->Xpix_per_m = arg.Xpix_per_m;
        this->Ypix_per_m = arg.Ypix_per_m;
        this->pixlen = arg.pixlen;
        pixels = new uint8_t[pixlen];
        for (int i = 0; i < pixlen; i++)
        {
            pixels[i] = arg.pixels[i];
        }
    }
    BMP_Image& operator=(const BMP_Image& arg)
    {
        this->bits_per_pixel = arg.bits_per_pixel;
        this->bytes_shift = arg.bytes_shift;
        this->colors_important = arg.colors_important;
        this->colors_used = arg.colors_used;
        this->compression = arg.compression;
        this->file_size = arg.file_size;
        this->header_size = arg.header_size;
        this->height = arg.height;
        this->width = arg.width;
        this->image_size = arg.image_size;
        this->signature = arg.signature;
        this->reserved_1 = arg.reserved_1;
        this->reserved_2 = arg.reserved_2;
        this->planes = arg.planes;
        this->Xpix_per_m = arg.Xpix_per_m;
        this->Ypix_per_m = arg.Ypix_per_m;
        this->pixlen = arg.pixlen;
        delete[] pixels;
        pixels = new uint8_t[pixlen];
        for (int i = 0; i < pixlen; i++)
        {
            pixels[i] = arg.pixels[i];
        }
        return *this;
    }
    int load(string filename)
    {
        if (filename.find(".bmp") == string::npos)
        {
            cout << "Error: unsupported file format. " << endl;
            return 0;
        }
        ifstream source_file;
        source_file.open(filename, ios::binary | ios::in);
        if (!source_file.is_open())
        {
            cout << "Error occured while tried to open image. " << endl;
            return 0;
        }
        // Reading header
        source_file.read((char*)&signature, sizeof(signature));
        source_file.read((char*)&file_size, sizeof(file_size));
        source_file.read((char*)&reserved_1, sizeof(reserved_1));
        source_file.read((char*)&reserved_2, sizeof(reserved_2));
        source_file.read((char*)&bytes_shift, sizeof(bytes_shift));
        // Reading info
        source_file.read((char*)&header_size, sizeof(header_size));
        source_file.read((char*)&width, sizeof(width));
        source_file.read((char*)&height, sizeof(height));
        source_file.read((char*)&planes, sizeof(planes));
        source_file.read((char*)&bits_per_pixel, sizeof(bits_per_pixel));
        source_file.read((char*)&compression, sizeof(compression));
        source_file.read((char*)&image_size, sizeof(image_size));
        source_file.read((char*)&Xpix_per_m, sizeof(Xpix_per_m));
        source_file.read((char*)&Ypix_per_m, sizeof(Ypix_per_m));
        source_file.read((char*)&colors_used, sizeof(colors_used));
        source_file.read((char*)&colors_important, sizeof(colors_important));

        // Reading pixels
        if (image_size == 0)
        {
            image_size = ((width * (bits_per_pixel / 8) + 3) & ~3) * height;
        }
        pixels = new uint8_t[width * height * bits_per_pixel / 8];
        pixlen = width * height * bits_per_pixel / 8;
        source_file.seekg(bytes_shift, source_file.beg);
        if (width % 4 == 0)
        {
            source_file.read((char*)pixels, image_size);
        }
        else
        {
            int row_stride = width * bits_per_pixel / 8;
            int new_stride = row_stride;
            while (new_stride % 4 != 0)
            {
                new_stride++;
            }
            //uint8_t* padding = new uint8_t[new_stride - row_stride]; //useless?
            int /*row = 0, useless ? */ pos = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < row_stride; x++)
                {
                    uint8_t tmp;
                    source_file >> tmp;
                    pixels[pos] = tmp;
                    pos++;
                }
                for (int x = 0; x < new_stride - row_stride; x++)
                {
                    uint8_t tmp_p;
                    source_file >> tmp_p;
                }
            }
        }
        source_file.close();
        return 1;
    }
    int save(string filename)
    {
        if (filename.find(".bmp") == string::npos)
        {
            cout << "Error: cannot save in such format. " << endl;
            return 0;
        }
        ofstream output_file;
        output_file.open(filename, ios::binary | ios::out);
        if (!output_file.is_open())
        {
            cout << "Error occured while tried to create *.bmp file. " << endl;
            return 0;
        }
        // Writing header
        output_file.write((char*)&signature, sizeof(signature));
        output_file.write((char*)&file_size, sizeof(file_size));
        output_file.write((char*)&reserved_1, sizeof(reserved_1));
        output_file.write((char*)&reserved_2, sizeof(reserved_2));
        output_file.write((char*)&bytes_shift, sizeof(bytes_shift));
        // Writing info
        output_file.write((char*)&header_size, sizeof(header_size));
        output_file.write((char*)&width, sizeof(width));
        output_file.write((char*)&height, sizeof(height));
        output_file.write((char*)&planes, sizeof(planes));
        output_file.write((char*)&bits_per_pixel, sizeof(bits_per_pixel));
        output_file.write((char*)&compression, sizeof(compression));
        output_file.write((char*)&image_size, sizeof(image_size));
        output_file.write((char*)&Xpix_per_m, sizeof(Xpix_per_m));
        output_file.write((char*)&Ypix_per_m, sizeof(Ypix_per_m));
        output_file.write((char*)&colors_used, sizeof(colors_used));
        output_file.write((char*)&colors_important, sizeof(colors_important));
        // Writing pixel data
        if (width % 4 == 0)
        {
            for (int i = 0; i < pixlen; i++)
            {
                output_file << pixels[i];
            }
        }
        else
        {
            int row_stride = width * bits_per_pixel / 8;
            int new_stride = row_stride;
            while (new_stride % 4 != 0)
            {
                new_stride++;
            }
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < row_stride; x++)
                {
                    output_file << pixels[y * row_stride + x];
                }
                for (int x_ = 0; x_ < new_stride - row_stride; x_++)
                {
                    output_file << 0;
                }
            }
        }
        output_file.close();
        return 1;
    }

    BMP_Image(int image_width, int image_height)
    {
        if (image_width <= 0 || image_height <= 0)
        {
            cout << "Error: incorrect image size. " << endl;
            exit(-1);
        }
        width = static_cast<uint32_t>(image_width);
        height = static_cast<uint32_t>(image_height);
        image_size = static_cast<uint32_t>(image_size = ((width * (bits_per_pixel / 8) + 3) & ~3) * height);
        file_size = 54 + image_size; /// ------- ! -------
        pixlen = width * height * bits_per_pixel / 8;
        pixels = new uint8_t[pixlen];
        // Filling with zeros
        for (int i = 0; i < pixlen; i++)
        {
            pixels[i] = 0;
        }
    }

    Pixel get_pixel(int x, int y)
    {
        int row_stride = width * bits_per_pixel / 8;
        if (x < 0 || y < 0 || 3 * x + y * row_stride >= pixlen)
        {
            cout << "Warning: incorrect pixel position. Returning empty pixel. " << endl;
            return Pixel(0, 0, 0, 0, 0);
        }
        int B = pixels[(height - y - 1) * row_stride + 3 * x];
        int G = pixels[(height - y - 1) * row_stride + 3 * x + 1];
        int R = pixels[(height - y - 1) * row_stride + 3 * x + 2];
        return Pixel(x, y, R, G, B);
    }
    int put_pixel(int x, int y, int R, int G, int B)
    {
        int row_stride = width * bits_per_pixel / 8;
        if (x < 0 || y < 0 || 3 * x + y * row_stride >= pixlen)
        {
            cout << "Error: incorrect pixel position. " << endl;
            return 0;
        }
        if (R < 0 || G < 0 || B < 0 || R > 255 || G > 255 || B > 255)
        {
            cout << "Error: incorrect color parameters. " << endl;
            return 0;
        }
        pixels[(height - y - 1) * row_stride + 3 * x] = B;
        pixels[(height - y - 1) * row_stride + 3 * x + 1] = G;
        pixels[(height - y - 1) * row_stride + 3 * x + 2] = R;
        return 1;
    }

    int get_height() const
    {
        return height;
    }
    int get_width() const
    {
        return width;
    }
    int get_file_size() const
    {
        return file_size;
    }
    int get_shift() const
    {
        return bytes_shift;
    }
    int get_datasize() const
    {
        return pixlen;
    }
    void print_params() const
    {
        cout << "Width " << width << endl;
        cout << "Height " << height << endl;
        cout << "File size " << file_size << endl;
        cout << "Image size " << image_size << endl;
        cout << "Bit depth " << bits_per_pixel << endl;
        cout << "Bytes shift " << bytes_shift << endl;
        cout << "Planes " << planes << endl;
        cout << "Compression code " << compression << endl;
    }
};

class Canvas
{
private:
    BMP_Image image;
    int width, height;
public:
    Canvas()
    {
        width = 0;
        height = 0;
        BMP_Image img(5, 5);
        image = img;
    }
    ~Canvas() {}
    Canvas(int x, int y)
    {
        BMP_Image tmp_image(x, y);
        image = tmp_image;
        width = image.get_width();
        height = image.get_height();
    }
    Canvas(string filename)
    {
        BMP_Image tmp_image(filename);
        image = tmp_image;
        width = image.get_width();
        height = image.get_height();
    }
    Canvas(const Canvas& arg)
    {
        width = arg.width;
        height = arg.height;
        image = arg.image;
    }
    int save(string filename)
    {
        return image.save(filename);
    }
    Canvas& operator=(const Canvas& arg)
    {
        width = arg.width;
        height = arg.height;
        image = arg.image;
        return *this;
    }
    int put_pixel(int x, int y, int R, int G, int B)
    {
        return image.put_pixel(x, y, R, G, B);
    }
    Pixel get_pixel(int x, int y)
    {
        return image.get_pixel(x, y);
    }
    int get_width()
    {
        return width;
    }
    int get_height()
    {
        return height;
    }
    int fill_canvas(int R, int G, int B) // Fills all canvas
    {
        if (R < 0 || G < 0 || B < 0 || R >= 256 || G >= 256 || B >= 256)
        {
            cout << "Error: incorrect RGB color parameters. " << endl;
            return 0;
        }
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (!put_pixel(x, y, R, G, B))
                {
                    cout << "Error occured while tried to fill canvas. " << endl;
                    return 0;
                }
            }
        }
        return 1;
    }
    int fill_area(int x, int y, int R, int G, int B, int R_stop = -1, int G_stop = -1, int B_stop = -1)
    {
        if (x < 0 || y < 0 || x >= width || y >= height)
        {
            cout << "Error: incorrect pixel position. " << endl;
            return 0;
        }
        if (R < 0 || G < 0 || B < 0 || R >= 256 || G >= 256 || B >= 256)
        {
            cout << "Error: incorrect RGB color parameters. " << endl;
            return 0;
        }
        if (R_stop != -1)
        {
            if (G_stop == -1 || B_stop == -1)
            {
                cout << "Error: 3 stop color params should be passed. " << endl;
                return 0;
            }
            else
            {
                if (R_stop < 0 || G_stop < 0 || B_stop < 0
                    || R_stop >= 256 || G_stop >= 256 || B_stop >= 256)
                {
                    cout << "Error: incorrect stop color params. " << endl;
                    return 0;
                }
            }
        }
        Pixel* last_painted_pixels = new Pixel[1];
        int last_painted_count = 1;
        int neighbour_count = 0;
        Pixel start_pixel = get_pixel(x, y);
        put_pixel(x, y, R, G, B);
        Pixel stop_pixel(0, 0, R_stop, G_stop, B_stop);
        Pixel curr_pixel(0, 0, R, G, B);
        bool fill_flag = true;
        last_painted_pixels[0] = get_pixel(x, y);
        while (fill_flag == true)
        {
            // Checking neighbours of last painted pixels
            neighbour_count = 0;
            vector<Pixel> curr_neighbours;
            curr_neighbours.clear();
            if (last_painted_count > 10e6)
            {
                cout << "Mem error" << endl;
                exit(-2);
            }
            for (int i = 0; i < last_painted_count; i++)
            {
                Pixel n_1(0, 0, 0, 0, 0), n_2(0, 0, 0, 0, 0), n_3(0, 0, 0, 0, 0),             // 1 2 3
                    n_4(0, 0, 0, 0, 0), n_5(0, 0, 0, 0, 0), n_6(0, 0, 0, 0, 0), n_7(0, 0, 0, 0, 0),   // 4 p 5
                    n_8(0, 0, 0, 0, 0), tmp;                                              // 6 7 8

                int valid_neighbours = 0;
                if (last_painted_pixels[i].x - 1 >= 0) // neighbour 4
                {
                    tmp = get_pixel(last_painted_pixels[i].x - 1, last_painted_pixels[i].y);
                    if (tmp != stop_pixel && tmp != curr_pixel)
                    {
                        if (R_stop != -1)
                        {
                            n_4 = tmp;
                            put_pixel(last_painted_pixels[i].x - 1, last_painted_pixels[i].y, R, G, B);
                            valid_neighbours++;
                            curr_neighbours.push_back(n_4);
                        }
                        else
                        {
                            if (tmp == start_pixel)
                            {
                                n_4 = tmp;
                                put_pixel(last_painted_pixels[i].x - 1, last_painted_pixels[i].y, R, G, B);
                                valid_neighbours++;
                                curr_neighbours.push_back(n_4);
                            }
                        }
                    }
                }
                if (last_painted_pixels[i].x + 1 < width) // neighbour 5
                {
                    tmp = get_pixel(last_painted_pixels[i].x + 1, last_painted_pixels[i].y);
                    if (tmp != stop_pixel && tmp != curr_pixel)
                    {
                        if (R_stop != -1)
                        {
                            n_5 = tmp;
                            put_pixel(last_painted_pixels[i].x + 1, last_painted_pixels[i].y, R, G, B);
                            valid_neighbours++;
                            curr_neighbours.push_back(n_5);
                        }
                        else
                        {
                            if (tmp == start_pixel)
                            {
                                n_5 = tmp;
                                put_pixel(last_painted_pixels[i].x + 1, last_painted_pixels[i].y, R, G, B);
                                valid_neighbours++;
                                curr_neighbours.push_back(n_5);
                            }
                        }
                    }
                }
                if (last_painted_pixels[i].y - 1 >= 0) // neighbour 2
                {
                    tmp = get_pixel(last_painted_pixels[i].x, last_painted_pixels[i].y - 1);
                    if (tmp != stop_pixel && tmp != curr_pixel)
                    {
                        if (R_stop != -1)
                        {
                            n_2 = tmp;
                            put_pixel(last_painted_pixels[i].x, last_painted_pixels[i].y - 1, R, G, B);
                            valid_neighbours++;
                            curr_neighbours.push_back(n_2);
                        }
                        else
                        {
                            if (tmp == start_pixel)
                            {
                                n_2 = tmp;
                                put_pixel(last_painted_pixels[i].x, last_painted_pixels[i].y - 1, R, G, B);
                                valid_neighbours++;
                                curr_neighbours.push_back(n_2);
                            }
                        }
                    }
                }
                if (last_painted_pixels[i].y + 1 < height) // neighbour 7
                {
                    tmp = get_pixel(last_painted_pixels[i].x, last_painted_pixels[i].y + 1);
                    if (tmp != stop_pixel && tmp != curr_pixel)
                    {
                        if (R_stop != -1)
                        {
                            n_7 = tmp;
                            put_pixel(last_painted_pixels[i].x, last_painted_pixels[i].y + 1, R, G, B);
                            valid_neighbours++;
                            curr_neighbours.push_back(n_7);
                        }
                        else
                        {
                            if (tmp == start_pixel)
                            {
                                n_7 = tmp;
                                put_pixel(last_painted_pixels[i].x, last_painted_pixels[i].y + 1, R, G, B);
                                valid_neighbours++;
                                curr_neighbours.push_back(n_7);
                            }
                        }
                    }
                }
                if (last_painted_pixels[i].x - 1 >= 0 && last_painted_pixels[i].y - 1 >= 0) // neighbour 1
                {

                    tmp = get_pixel(last_painted_pixels[i].x - 1, last_painted_pixels[i].y - 1);
                    Pixel tmp_right = get_pixel(tmp.x + 1, tmp.y);
                    Pixel tmp_bottom = get_pixel(tmp.x, tmp.y + 1);
                    if ((tmp_right != stop_pixel && tmp_bottom != stop_pixel) &&
                        (tmp_right == start_pixel && tmp_bottom == start_pixel)) {
                        if (tmp != stop_pixel && tmp != curr_pixel)
                        {
                            if (R_stop != -1)
                            {
                                n_1 = tmp;
                                put_pixel(last_painted_pixels[i].x - 1, last_painted_pixels[i].y - 1, R, G, B);
                                valid_neighbours++;
                                curr_neighbours.push_back(n_1);
                            }
                            else
                            {
                                if (tmp == start_pixel)
                                {
                                    n_1 = tmp;
                                    put_pixel(last_painted_pixels[i].x - 1, last_painted_pixels[i].y - 1, R, G, B);
                                    valid_neighbours++;
                                    curr_neighbours.push_back(n_1);
                                }
                            }
                        }
                    }
                }
                if (last_painted_pixels[i].x + 1 < width && last_painted_pixels[i].y - 1 >= 0) // neighbour 3
                {
                    tmp = get_pixel(last_painted_pixels[i].x + 1, last_painted_pixels[i].y - 1);
                    Pixel tmp_left = get_pixel(tmp.x - 1, tmp.y);
                    Pixel tmp_bottom = get_pixel(tmp.x, tmp.y + 1);
                    if ((tmp_left != stop_pixel && tmp_bottom != stop_pixel) &&
                        (tmp_left == start_pixel && tmp_bottom == start_pixel)) {
                        if (tmp != stop_pixel && tmp != curr_pixel)
                        {
                            if (R_stop != -1)
                            {
                                n_3 = tmp;
                                put_pixel(last_painted_pixels[i].x + 1, last_painted_pixels[i].y - 1, R, G, B);
                                valid_neighbours++;
                                curr_neighbours.push_back(n_3);
                            }
                            else
                            {
                                if (tmp == start_pixel)
                                {
                                    n_3 = tmp;
                                    put_pixel(last_painted_pixels[i].x + 1, last_painted_pixels[i].y - 1, R, G, B);
                                    valid_neighbours++;
                                    curr_neighbours.push_back(n_3);
                                }
                            }
                        }
                    }
                }
                if (last_painted_pixels[i].x - 1 >= 0 && last_painted_pixels[i].y + 1 < height) // neighbour 6
                {
                    tmp = get_pixel(last_painted_pixels[i].x - 1, last_painted_pixels[i].y + 1);
                    Pixel tmp_upper = get_pixel(tmp.x, tmp.y - 1);
                    Pixel tmp_right = get_pixel(tmp.x + 1, tmp.y);
                    if ((tmp_upper != stop_pixel && tmp_right != stop_pixel) &&
                        (tmp_upper == start_pixel && tmp_right == start_pixel)) {
                        if (tmp != stop_pixel && tmp != curr_pixel)
                        {
                            if (R_stop != -1)
                            {
                                n_6 = tmp;
                                put_pixel(last_painted_pixels[i].x - 1, last_painted_pixels[i].y + 1, R, G, B);
                                valid_neighbours++;
                                curr_neighbours.push_back(n_6);
                            }
                            else
                            {
                                if (tmp == start_pixel)
                                {
                                    n_6 = tmp;
                                    put_pixel(last_painted_pixels[i].x - 1, last_painted_pixels[i].y + 1, R, G, B);
                                    valid_neighbours++;
                                    curr_neighbours.push_back(n_6);
                                }
                            }
                        }
                    }
                }
                if (last_painted_pixels[i].x + 1 < width && last_painted_pixels[i].y + 1 < height) // neighbour 8
                {
                    tmp = get_pixel(last_painted_pixels[i].x + 1, last_painted_pixels[i].y + 1);
                    Pixel tmp_left = get_pixel(tmp.x - 1, tmp.y);
                    Pixel tmp_upper = get_pixel(tmp.x, tmp.y - 1);
                    if ((tmp_left != stop_pixel && tmp_upper != stop_pixel) &&
                        (tmp_left == start_pixel && tmp_upper == start_pixel)) {
                        if (tmp != stop_pixel && tmp != curr_pixel)
                        {
                            if (R_stop != -1)
                            {
                                n_8 = tmp;
                                put_pixel(last_painted_pixels[i].x + 1, last_painted_pixels[i].y + 1, R, G, B);
                                valid_neighbours++;
                                curr_neighbours.push_back(n_8);
                            }
                            else
                            {
                                if (tmp == start_pixel)
                                {
                                    n_8 = tmp;
                                    put_pixel(last_painted_pixels[i].x + 1, last_painted_pixels[i].y + 1, R, G, B);
                                    valid_neighbours++;
                                    curr_neighbours.push_back(n_8);
                                }
                            }
                        }
                    }
                }
                // Painted and saved all valid neighbours.
                neighbour_count += valid_neighbours;
            }
            if (neighbour_count == 0)
            {
                fill_flag = false;
            }
            delete[] last_painted_pixels;
            last_painted_pixels = new Pixel[neighbour_count];
            for (int j = 0; j < static_cast<int>(curr_neighbours.size()); j++)
            {
                last_painted_pixels[j] = curr_neighbours[j];
            }
            last_painted_count = static_cast<int>(curr_neighbours.size());
        }
        return 1;
    }
    int draw_segment(int x1, int y1, int x2, int y2, int R, int G, int B)
    {
        if (x1 < 0 || y1 < 0 || x2 < 0 || y2 < 0
            || x1 >= width || y1 >= height || x2 >= width || y2 >= height)
        {
            cout << "Error: incorrect segment verticies positions. " << endl;
            return 0;
        }
        if (R < 0 || G < 0 || B < 0 || R >= 256 || G >= 256 || B >= 256)
        {
            cout << "Error: incorrect RGB color parameters. " << endl;
            return 0;
        }
        if (x1 == x2 && y1 == y2)
        {
            put_pixel(x1, y1, R, G, B);
        }
        if (x1 == x2 && y1 != y2)
        {
            for (int i = 0; i < max(y1, y2) - min(y1, y2); i++)
            {
                put_pixel(x1, min(y1, y2) + i, R, G, B);
            }
        }
        if (x1 != x2 && y1 == y2)
        {
            for (int i = 0; i < max(x1, x2) - min(x1, x2); i++)
            {
                put_pixel(min(x1, x2) + i, y1, R, G, B);
            }
        }
        if (x1 != x2 && y1 != y2)
        {
            if (x1 > x2)
            {
                int tmp; tmp = x1;
                x1 = x2; x2 = tmp;
                tmp = y1; y1 = y2;
                y2 = tmp;
            }
            int delta_x = x2 - x1;
            int delta_y = max(y1, y2) - min(y1, y2);
            double error = 0.0;
            double delta_error = static_cast<double>(delta_y + 1) / static_cast<double>(delta_x + 1);
            if (delta_y <= delta_x) {
                int y = y1;
                for (int x = x1; x < x2 + 1; x++)
                {
                    if (!put_pixel(x, y, R, G, B))
                    {
                        return 0;
                    }
                    error += delta_error;
                    if (error >= 1.0)
                    {
                        y += (y2 > y1) ? 1 : -1;
                        error -= 1.0;
                    }
                }
            }
            else
            {
                int x = x1;
                //int y = y1; useless?
                delta_error = static_cast<double>(delta_x + 1) / static_cast<double>(delta_y + 1);
                for (int delta_y = 0; delta_y < max(y1, y2) - min(y1, y2); delta_y++)
                {
                    if (!put_pixel(x, (y1 > y2) ? y1 - delta_y : y1 + delta_y, R, G, B))
                    {
                        return 0;
                    }
                    error += delta_error;
                    if (error >= 1.0)
                    {
                        x += 1;
                        error -= 1.0;
                    }
                }
            }
        }
        return 1;
    }
    int draw_segment(int x1, int y1, int x2, int y2, int w, int R, int G, int B)
    {
        if (x1 < 0 || y1 < 0 || x2 < 0 || y2 < 0 ||
            x1 >= width || y1 >= height || x2 >= width || y2 >= height)
        {
            cout << "Error: incorrect segment verticies positions. (w =  " << w << ")" << endl;
            return 0;
        }
        if (R < 0 || G < 0 || B < 0 || R >= 256 || G >= 256 || B >= 256)
        {
            cout << "Error: incorrect RGB color parameters. " << endl;
            return 0;
        }
        if (width <= 0 || x1 - w < 0 || x1 + w >= width || x2 - w < 0 || x2 + w >= width ||
            y1 - w < 0 || y1 + 2 >= height || y2 - w < 0 || y2 + w >= height)
        {
            cout << "Error: incorrect width parameter. " << endl;
            return 0;
        }
        if (w == 1)
        {
            return draw_segment(x1, y1, x2, y2, R, G, B);
        }
        if (x1 != x2)
        {
            for (int i = 0; i < w; i++)
            {
                int y_plus;
                (i % 2 == 0) ? (y_plus = -(i / 2)) : (y_plus = (i + 1) / 2);
                int draw_status = draw_segment(x1, y1 + y_plus, x2, y2 + y_plus, R, G, B);
                if (!draw_status)
                {
                    cout << "Error occured while tried to draw segment with width = " << w << endl;
                    return 0;
                }
            }
        }
        else
        {
            for (int i = 0; i < w; i++)
            {
                int x_plus;
                (i % 2 == 0) ? (x_plus = -i / 2) : (x_plus = (i + 1) / 2);
                if (!draw_segment(x1 + x_plus, y1, x2 + x_plus, y2, R, G, B))
                {
                    return 0;
                }
            }
        }
        return 1;
    }
    int draw_rectangle(int x1, int y1, int x2, int y2, int R, int G, int B, int w = 1, int R_f = -1, int G_f = -1, int B_f = -1)
    {
        if (x1 < 0 || x1 >= width || y1 < 0 || y1 >= height || x2 < 0 || x2 >= width || y2 < 0 || y2 >= height)
        {
            cout << "Error: incorrect rectangle verticies positions. " << endl;
            return 0;
        }
        if (w != 1)
        {
            if (w <= 0 || x1 - w / 2 < 0 || x1 + w / 2 >= width || y1 - w / 2 < 0 || y1 + w / 2 >= height ||
                x2 - w / 2 < 0 || x2 + w / 2 >= width || y2 - w / 2 < 0 || y2 + w / 2 >= height)
            {
                cout << "Error: incorrect width parameter: rectangle goes out of the image border. " << endl;
                return 0;
            }
        }
        if (R_f != -1)
        {
            if (G_f == -1 || B_f == -1)
            {
                cout << "Error: 3 fill color parameters should be passed. " << endl;
                return 0;
            }
            else
            {
                if (R_f < 0 || G_f < 0 || B_f < 0 || R_f >= 256 || G_f >= 256 || B_f >= 256)
                {
                    cout << "Error: incorrect RGB fill color parameters. " << endl;
                    return 0;
                }
            }
        }
        if (R < 0 || G < 0 || B < 0 || R >= 256 || G >= 256 || B >= 256)
        {
            cout << "Error: incorrect RGB border color parameters. " << endl;
            return 0;
        }
        int x3 = x1 + max(x1, x2) - min(x1, x2);
        int y3 = y1;
        int x4 = x1;
        int y4 = y1 + max(y1, y2) - min(y1, y2);
        for (int i = 0; i < w; i++)
        {
            if (!draw_segment(x1 + i, y1 + i, x3 - i, y3 + i, R, G, B) ||
                !draw_segment(x1 + i, y1 + i, x4 + i, y4 - i, R, G, B) ||
                !draw_segment(x3 - i, y3 + i, x2 - i, y2 - i + 1, R, G, B) ||
                !draw_segment(x4 + i, y4 - i, x2 - i, y2 - i, R, G, B))
            {
                cout << "Error occured while tried to draw rectangle`s side. " << endl;
                return 0;
            }
        }
        // Filling if needed
        if (R_f != -1 && G_f != -1 && B_f != -1)
        {
            for (int x = min(x1, x2) + w; x < max(x1, x2) - w + 1; x++)
            {
                for (int y = min(y1, y2) + w; y < max(y1, y2) - w + 1; y++)
                {
                    if (!put_pixel(x, y, R_f, G_f, B_f))
                    {
                        cout << "Error occured while tried to fill area inside rectangle. " << endl;
                        return 0;
                    }
                }
            }
        }
        return 1;
    }
    int draw_triangle(int x1, int y1, int x2, int y2, int x3, int y3, int R, int G, int B, int R_f = -1, int G_f = -1, int B_f = -1)
    {
        if (x1 < 0 || y1 < 0 ||
            x2 < 0 || y2 < 0 ||
            x3 < 0 || y3 < 0 ||
            x1 >= width || y1 >= height ||
            x2 >= width || y2 >= height ||
            x3 >= width || y3 >= height)
        {
            cout << "Error: incorrect triangle verticies position. " << endl;
            return 0;
        }
        if (R < 0 || G < 0 || B < 0 ||
            R >= 256 || G >= 256 || B >= 256)
        {
            cout << "Error: incorrect RGB color params. " << endl;
            return 0;
        }
        if (R_f != -1)
        {
            if (G_f == -1 || B_f == -1)
            {
                cout << "Error: 3 RGB fill color params should be passed. " << endl;
                return 0;
            }
            else
            {
                if (R_f < 0 || G_f < 0 || B_f < 0 ||
                    R_f >= 256 || G_f >= 256 || B_f >= 256)
                {
                    cout << "Error: incorrect fill color params. " << endl;
                    return 0;
                }
            }
        }
        if (!draw_segment(x1, y1, x2, y2, R, G, B) ||
            !draw_segment(x2, y2, x3, y3, R, G, B) ||
            !draw_segment(x1, y1, x3, y3, R, G, B))
        {
            cout << "Error: cannot draw one of sides of triangle. " << endl;
            return 0;
        }
        // Filling if needed
        if (R_f != -1)
        {
            int inner_vect_x;
            int inner_vect_y;
            inner_vect_x = ((x1 - x3) + (x2 - x3)) / 4;
            inner_vect_y = ((y1 - y3) + (y2 - y3)) / 4;
            fill_area(x3 + inner_vect_x, y3 + inner_vect_y, R_f, G_f, B_f, R, G, B);
        }
        return 1;
    }
    int draw_circle(int x, int y, int Rad, int R, int G, int B, int R_f = -1, int G_f = -1, int B_f = -1)
    {
        if (x < 0 || y < 0 || x >= width || y >= width)
        {
            cout << "Error: incorrect circle center position. " << endl;
            return 0;
        }
        if (Rad <= 0 || Rad >= width || Rad >= height)
        {
            cout << "Error: incorrect circle radius parameter. " << endl;
            return 0;
        }
        if (R < 0 || G < 0 || B < 0 ||
            R >= 256 || G >= 256 || B >= 256)
        {
            cout << "Error: incorrect RGB border color parameters." << endl;
            return 0;
        }
        if (R_f != -1)
        {
            if (G_f == -1 || B_f == -1)
            {
                cout << "Error: 3 RGB fill color parameters should be passed. " << endl;
                return 0;
            }
            else
            {
                if (R_f < 0 || G_f < 0 || B_f < 0
                    || R_f >= 256 || G_f >= 256 || B_f >= 256)
                {
                    cout << "Error: incorrect RGB fill color parameters. " << endl;
                    return 0;
                }
            }
        }
        int x_ = 0;
        int y_ = Rad;
        int gap = 0;
        int delta = 2 * (1 - Rad);
        while (y_ >= 0)
        {
            if (!put_pixel(x + x_, y + y_, R, G, B) ||
                !put_pixel(x + x_, y - y_, R, G, B) ||
                !put_pixel(x - x_, y + y_, R, G, B) ||
                !put_pixel(x - x_, y - y_, R, G, B))
            {
                cout << "Error occured while tried to draw a circle. " << endl;
                return 0;
            }
            gap = 2 * (delta + y_) - 1;
            if (delta < 0 && gap <= 0)
            {
                x_++;
                delta += 2 * x_ + 1;
                continue;
            }
            if (delta > 0 && gap > 0)
            {
                y_--;
                delta -= 2 * y_ + 1;
                continue;
            }
            x_++;
            delta += 2 * (x_ - y_);
            y_--;
        }
        // Filling if needed
        if (R_f != -1)
        {
            fill_area(x, y, R_f, G_f, B_f, R, G, B);
        }
        return 1;
    }
    int draw_circle(int x, int y, int Rad, int R, int G, int B, int w, int R_f = -1, int G_f = -1, int B_f = -1)
    {
        for (int i = 0; i < w; i++)
        {
            if (!draw_circle(x, y, Rad - i, R, G, B))
            {
                cout << "Error occured while tried to draw a circle (w = " << w << ") " << endl;
                return 0;
            }
        }
        // Filling if needed
        if (R_f != -1)
        {
            if (G_f == -1 || B_f == -1)
            {
                cout << "Error: 3 RGB fill color parameters should be passed. " << endl;
                return 0;
            }
            else
            {
                if (R_f < 0 || G_f < 0 || B_f < 0 ||
                    R_f >= 256 || G_f >= 256 || B_f >= 256)
                {
                    cout << "Error: incorrect RGB fill color parameters. " << endl;
                    return 0;
                }
                fill_area(x, y, R_f, G_f, B_f, R, G, B);
            }
        }
        return 1;
    }
    int plot(double* var_values, double* func_values, int sample_count)
    {
        cout << "Warning: calling default plot function will clear canvas." << endl;
        this->image = BMP_Image(800, 800);
        this->width = 800;
        this->height = 800;
        // Clearing canvas
        this->fill_canvas(255, 255, 255);
        if (var_values == nullptr || func_values == nullptr)
        {
            cout << "Error: passed empty arrays." << endl;
            return 0;
        }
        if (sample_count <= 0)
        {
            cout << "Error: incorrect array length passed. " << endl;
            return 0;
        }
        // Sorting
        double* new_var = new double[sample_count];
        double* new_func = new double[sample_count];
        for (int i = 0; i < sample_count; i++)
        {
            new_var[i] = var_values[i];
            new_func[i] = func_values[i];
        }
        sort(new_var, new_var + sample_count);
        sort(new_func, new_func + sample_count);
        // Drawing borders
        draw_rectangle(3, 3, 797, 797, 120, 120, 120);
        // Calculating variable and value borders, locating var zero and func zeros.
        double min_x = var_values[0], max_x = var_values[0], min_f = func_values[0], max_f = func_values[0];
        double zero_proportion_x = -1.0, zero_proportion_f = -1.0;
        for (int i = 0; i < sample_count; i++)
        {
            if (new_var[i] < min_x)
            {
                min_x = new_var[i];
            }
            if (new_var[i] > max_x)
            {
                max_x = new_var[i];
            }
            if (new_func[i] < min_f)
            {
                min_f = new_func[i];
            }
            if (new_func[i] > max_f)
            {
                max_f = new_func[i];
            }
            if (i == 0)
            {
                if (new_var[i] == 0)
                {
                    zero_proportion_x = 0;
                }
                if (new_func[i] == 0)
                {
                    zero_proportion_f = 0;
                }
            }

            if (i != 0)
            {
                if (new_var[i] == 0)
                {
                    zero_proportion_x = abs(new_var[0]) / abs(new_var[sample_count - 1] - new_var[0]);
                }
                if (func_values[i] == 0)
                {
                    zero_proportion_f = abs(new_func[0]) / abs(new_func[sample_count - 1] - new_func[0]);
                }
                if (var_values[i - 1] * var_values[i] < 0)
                {
                    zero_proportion_x = abs(new_var[0]) / abs(new_var[sample_count - 1] - new_var[0]);
                }
                if (func_values[i - 1] * func_values[i] < 0)
                {
                    zero_proportion_f = abs(new_func[0]) / abs(new_func[sample_count - 1] - new_func[0]);
                }
            }
        }
        // Drawing axes
        // 1. if plot is located in I/II/III/IV quarters
        if (zero_proportion_x == -1.0 && zero_proportion_f == -1.0)
        {
            // A. I-st quarter
            if (min_x >= 0 && min_f >= 0)
            {
                draw_segment(10, 790, 790, 790, 0, 0, 0); // Ox
                draw_segment(10, 10, 10, 790, 0, 0, 0); // Oy
            }
            // B. II-nd quarter
            if (max_x <= 0 && min_f >= 0)
            {
                draw_segment(10, 790, 790, 790, 0, 0, 0); // Ox
                draw_segment(790, 10, 790, 790, 0, 0, 0); // Oy
            }
            // C. III-rd quarter
            if (max_x <= 0 && max_f <= 0)
            {
                draw_segment(10, 10, 790, 10, 0, 0, 0); // Ox
                draw_segment(790, 10, 790, 790, 0, 0, 0); // Oy
            }
            // D. IV-th querter
            if (min_x >= 0 && max_f <= 0)
            {
                draw_segment(10, 10, 790, 10, 0, 0, 0); // Ox
                draw_segment(10, 10, 10, 790, 0, 0, 0);
            }
        }
        // 2. if plot intersects Y axis and dont intersect X axis
        if (zero_proportion_f == -1.0 && zero_proportion_x != -1.0)
        {
            int Y_axis = 10 + static_cast<int>(zero_proportion_x * 780.0);
            // A. I-II quarters
            if (min_f >= 0)
            {
                draw_segment(10, 790, 790, 790, 0, 0, 0); // Ox
                draw_segment(Y_axis, 10, Y_axis, 790, 0, 0, 0); // Oy
            }
            // B. III-IV quarters
            if (max_f <= 0)
            {
                draw_segment(10, 10, 790, 10, 0, 0, 0); // Ox
                draw_segment(Y_axis, 10, Y_axis, 790, 0, 0, 0); // Oy
            }
        }
        // 3. if plot intersects X axis and dont intersect Y axis
        if (zero_proportion_x == -1.0 && zero_proportion_f != -1.0)
        {
            int X_axis = 790 - static_cast<int>(zero_proportion_f * 780.0);
            // A. I-II quarters
            if (min_x >= 0)
            {
                draw_segment(10, X_axis, 790, X_axis, 0, 0, 0); // Ox
                draw_segment(10, 10, 10, 790, 0, 0, 0); // Oy

            }
            // B. III-IV quarters
            if (max_x <= 0)
            {
                draw_segment(10, X_axis, 790, X_axis, 0, 0, 0); // Ox
                draw_segment(790, 10, 790, 790, 0, 0, 0); // Oy
            }
        }
        // 4. if plot intersects both X & Y axes
        if (zero_proportion_x != -1.0 && zero_proportion_f != -1.0)
        {
            int Y_axis = 10 + static_cast<int>(zero_proportion_x * 780.0);
            int X_axis = 790 - static_cast<int>(zero_proportion_f * 780.0);
            draw_segment(10, X_axis, 790, X_axis, 0, 0, 0); // Ox
            draw_segment(Y_axis, 10, Y_axis, 790, 0, 0, 0); // Oy

        }
        // Values normalization
        double k_x;
        double k_f;
        if (zero_proportion_x != -1.0)
        {
            k_x = 778 / (max_x - min_x);
        }
        if (zero_proportion_f != -1.0)
        {
            k_f = 778 / (max_f - min_f);
        }
        if (zero_proportion_x == -1.0)
        {
            if (min_x >= 0)
            {
                k_x = 778 / (max_x);
            }
            if (max_x <= 0)
            {
                k_x = 778 / (-min_x);
            }
        }
        if (zero_proportion_f == -1.0)
        {
            if (min_f >= 0)
            {
                k_f = 778 / (max_f);
            }
            if (max_f <= 0)
            {
                k_f = 778 / (-min_f);
            }
        }
        double k = min(k_x, k_f);
        for (int i = 0; i < sample_count; i++)
        {
            var_values[i] = k * var_values[i];
            func_values[i] = k * func_values[i];
        }
        // Integer approximations
        int* x_appr = new int[sample_count];
        int* f_appr = new int[sample_count];
        int x_null = 10;
        int f_null = 790;
        if (zero_proportion_x != -1.0)
        {
            x_null += static_cast<int>(780 * zero_proportion_x);
        }
        if (zero_proportion_f != -1.0)
        {
            f_null -= static_cast<int>(780 * zero_proportion_f);
        }
        for (int i = 0; i < sample_count; i++)
        {
            int x = static_cast<int>(var_values[i]);
            int x_;
            int f = static_cast<int>(func_values[i]);
            int f_;
            x_ = x_null + x;
            if (f <= 0)
            {
                f_ = f_null + abs(f);
            }
            if (f > 0)
            {
                f_ = f_null - f;
            }
            x_appr[i] = x_;
            f_appr[i] = f_;
        }
        // Plotting
        for (int i = 0; i < sample_count - 1; i++)
        {
            if (!draw_segment(x_appr[i], f_appr[i], x_appr[i + 1], f_appr[i + 1], 3, 27, 27, 137))
            {
                cout << "Error occured while plotting. " << endl;
                return 0;
            }
        }
        return 1;
    }
    int scatterplot(double* var_values, double* func_values, int sample_count)
    {
        cout << "Warning: calling default scatterplot function will clear canvas." << endl;
        this->image = BMP_Image(800, 800);
        this->width = 800;
        this->height = 800;
        // Clearing canvas
        this->fill_canvas(255, 255, 255);
        if (var_values == nullptr || func_values == nullptr)
        {
            cout << "Error: passed empty arrays." << endl;
            return 0;
        }
        if (sample_count <= 0)
        {
            cout << "Error: incorrect array length passed. " << endl;
            return 0;
        }
        // Sorting
        double* new_var = new double[sample_count];
        double* new_func = new double[sample_count];
        for (int i = 0; i < sample_count; i++)
        {
            new_var[i] = var_values[i];
            new_func[i] = func_values[i];
        }
        sort(new_var, new_var + sample_count);
        sort(new_func, new_func + sample_count);
        // Drawing borders
        draw_rectangle(3, 3, 797, 797, 120, 120, 120);
        // Calculating variable and value borders, locating var zero and func zeros.
        double min_x = var_values[0], max_x = var_values[0], min_f = func_values[0], max_f = func_values[0];
        double zero_proportion_x = -1.0, zero_proportion_f = -1.0;
        for (int i = 0; i < sample_count; i++)
        {
            if (new_var[i] < min_x)
            {
                min_x = new_var[i];
            }
            if (new_var[i] > max_x)
            {
                max_x = new_var[i];
            }
            if (new_func[i] < min_f)
            {
                min_f = new_func[i];
            }
            if (new_func[i] > max_f)
            {
                max_f = new_func[i];
            }
            if (i == 0)
            {
                if (new_var[i] == 0)
                {
                    zero_proportion_x = 0;
                }
                if (new_func[i] == 0)
                {
                    zero_proportion_f = 0;
                }
            }

            if (i != 0)
            {
                if (new_var[i] == 0)
                {
                    zero_proportion_x = abs(new_var[0]) / abs(new_var[sample_count - 1] - new_var[0]);
                }
                if (func_values[i] == 0)
                {
                    zero_proportion_f = abs(new_func[0]) / abs(new_func[sample_count - 1] - new_func[0]);
                }
                if (var_values[i - 1] * var_values[i] < 0)
                {
                    zero_proportion_x = abs(new_var[0]) / abs(new_var[sample_count - 1] - new_var[0]);
                }
                if (func_values[i - 1] * func_values[i] < 0)
                {
                    zero_proportion_f = abs(new_func[0]) / abs(new_func[sample_count - 1] - new_func[0]);
                }
            }
        }
        // Drawing axes
        // 1. if plot is located in I/II/III/IV quarters
        if (zero_proportion_x == -1.0 && zero_proportion_f == -1.0)
        {
            // A. I-st quarter
            if (min_x >= 0 && min_f >= 0)
            {
                draw_segment(10, 790, 790, 790, 0, 0, 0); // Ox
                draw_segment(10, 10, 10, 790, 0, 0, 0); // Oy
            }
            // B. II-nd quarter
            if (max_x <= 0 && min_f >= 0)
            {
                draw_segment(10, 790, 790, 790, 0, 0, 0); // Ox
                draw_segment(790, 10, 790, 790, 0, 0, 0); // Oy
            }
            // C. III-rd quarter
            if (max_x <= 0 && max_f <= 0)
            {
                draw_segment(10, 10, 790, 10, 0, 0, 0); // Ox
                draw_segment(790, 10, 790, 790, 0, 0, 0); // Oy
            }
            // D. IV-th querter
            if (min_x >= 0 && max_f <= 0)
            {
                draw_segment(10, 10, 790, 10, 0, 0, 0); // Ox
                draw_segment(10, 10, 10, 790, 0, 0, 0);
            }
        }
        // 2. if plot intersects Y axis and dont intersect X axis
        if (zero_proportion_f == -1.0 && zero_proportion_x != -1.0)
        {
            int Y_axis = 10 + static_cast<int>(zero_proportion_x * 780.0);
            // A. I-II quarters
            if (min_f >= 0)
            {
                draw_segment(10, 790, 790, 790, 0, 0, 0); // Ox
                draw_segment(Y_axis, 10, Y_axis, 790, 0, 0, 0); // Oy
            }
            // B. III-IV quarters
            if (max_f <= 0)
            {
                draw_segment(10, 10, 790, 10, 0, 0, 0); // Ox
                draw_segment(Y_axis, 10, Y_axis, 790, 0, 0, 0); // Oy
            }
        }
        // 3. if plot intersects X axis and dont intersect Y axis
        if (zero_proportion_x == -1.0 && zero_proportion_f != -1.0)
        {
            int X_axis = 790 - static_cast<int>(zero_proportion_f * 780.0);
            // A. I-II quarters
            if (min_x >= 0)
            {
                draw_segment(10, X_axis, 790, X_axis, 0, 0, 0); // Ox
                draw_segment(10, 10, 10, 790, 0, 0, 0); // Oy

            }
            // B. III-IV quarters
            if (max_x <= 0)
            {
                draw_segment(10, X_axis, 790, X_axis, 0, 0, 0); // Ox
                draw_segment(790, 10, 790, 790, 0, 0, 0); // Oy
            }
        }
        // 4. if plot intersects both X & Y axes
        if (zero_proportion_x != -1.0 && zero_proportion_f != -1.0)
        {
            int Y_axis = 10 + static_cast<int>(zero_proportion_x * 780.0);
            int X_axis = 790 - static_cast<int>(zero_proportion_f * 780.0);
            draw_segment(10, X_axis, 790, X_axis, 0, 0, 0); // Ox
            draw_segment(Y_axis, 10, Y_axis, 790, 0, 0, 0); // Oy

        }
        // Values normalization
        double k_x;
        double k_f;
        if (zero_proportion_x != -1.0)
        {
            k_x = 778 / (max_x - min_x);
        }
        if (zero_proportion_f != -1.0)
        {
            k_f = 778 / (max_f - min_f);
        }
        if (zero_proportion_x == -1.0)
        {
            if (min_x >= 0)
            {
                k_x = 778 / (max_x);
            }
            if (max_x <= 0)
            {
                k_x = 778 / (-min_x);
            }
        }
        if (zero_proportion_f == -1.0)
        {
            if (min_f >= 0)
            {
                k_f = 778 / (max_f);
            }
            if (max_f <= 0)
            {
                k_f = 778 / (-min_f);
            }
        }
        double k = min(k_x, k_f);
        for (int i = 0; i < sample_count; i++)
        {
            var_values[i] = k * var_values[i];
            func_values[i] = k * func_values[i];
        }
        // Integer approximations
        int* x_appr = new int[sample_count];
        int* f_appr = new int[sample_count];
        int x_null = 10;
        int f_null = 790;
        if (zero_proportion_x != -1.0)
        {
            x_null += static_cast<int>(780 * zero_proportion_x);
        }
        if (zero_proportion_f != -1.0)
        {
            f_null -= static_cast<int>(780 * zero_proportion_f);
        }
        for (int i = 0; i < sample_count; i++)
        {
            int x = static_cast<int>(var_values[i]);
            int x_;
            int f = static_cast<int>(func_values[i]);
            int f_;
            x_ = x_null + x;
            if (f <= 0)
            {
                f_ = f_null + abs(f);
            }
            if (f > 0)
            {
                f_ = f_null - f;
            }
            x_appr[i] = x_;
            f_appr[i] = f_;
        }
        // Plotting
        for (int i = 0; i < sample_count; i++)
        {
            if (!draw_rectangle(x_appr[i] - 2, f_appr[i] - 2, x_appr[i] + 2, f_appr[i] + 2, 27, 27, 137, 1, 27, 27, 137))
            {
                cout << "Error occured while plotting. " << endl;
                return 0;
            }
        }
        return 1;
    }
    Canvas copy_fragment(int x1, int y1, int x2, int y2)
    {
        if (x1 < 0 || y1 < 0 || x2 < 0 || y2 < 0 ||
            x1 >= width || y1 >= height || x2 >= width || y2 >= height)
        {
            cout << "Error: incorrect location of area to copy." << endl;
            Canvas err(100, 100);
            return err;
        }
        if (x1 > x2)
        {
            int tmp;
            tmp = x1;
            x1 = x2;
            x2 = tmp;
            tmp = y1;
            y1 = y2;
            y2 = tmp;
        }
        Canvas temp(x2 - x1 + 1, max(y2, y1) - min(y2, y1) + 1);
        for (int x = 0; x <= x2 - x1; x++)
        {
            for (int y = 0; y <= max(y1, y2) - min(y1, y2);)
            {
                Pixel tmp = get_pixel(x1 + x, y1 + y);
                temp.put_pixel(x, y, tmp.R, tmp.G, tmp.B);
                if (y2 > y1)
                {
                    y++;
                }
                if (y2 < y1)
                {
                    y--;
                }
            }
        }
        return temp;
    }
    int insert_fragment(int x, int y, string filename)
    {
        if (filename.find(".bmp") == string::npos)
        {
            cout << "Error: incorrect file format passed in insertion function. " << endl;
            return 0;
        }
        if (x < 0 || y < 0 || x >= width || y >= height)
        {
            cout << "Error: incorrect insertion point." << endl;
            return 0;
        }
        Canvas insertion(filename);
        insertion.save("DEBUG_INSERT.bmp");
        if (insertion.width + x > width || insertion.height + y > height)
        {
            cout << "Error: cannot insert such fragment - its out of picture bounds." << endl;
            return 0;
        }
        for (int x_ = 0; x_ < insertion.width; x_++)
        {
            for (int y_ = 0; y_ < insertion.height; y_++)
            {
                Pixel ins_pix = insertion.get_pixel(x_, y_);
                if (!put_pixel(x + x_, y + y_, ins_pix.R, ins_pix.G, ins_pix.B))
                {
                    cout << "Error occured while tried to insert fragment." << filename << endl;
                    return 0;
                }
            }
        }
        return 1;
    }
    int insert_fragment(int x, int y, Canvas insertion)
    {
        if (x < 0 || y < 0 || x >= width || y >= height)
        {
            cout << "Error: incorrect insertion point." << endl;
            return 0;
        }
        if (insertion.width + x > width || insertion.height + y > height)
        {
            cout << "Error: cannot insert such fragment - its out of picture bounds." << endl;
            return 0;
        }
        for (int x_ = 0; x_ < insertion.width; x_++)
        {
            for (int y_ = 0; y_ < insertion.height; y_++)
            {
                Pixel ins_pix = insertion.get_pixel(x_, y_);
                if (!put_pixel(x + x_, y + y_, ins_pix.R, ins_pix.G, ins_pix.B))
                {
                    cout << "Error occured while tried to insert fragment." << endl;
                    return 0;
                }
            }
        }
        return 1;
    }
    int to_black_n_white()
    {
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                Pixel curr_pix = get_pixel(x, y);
                int mean_color = static_cast<int>((static_cast<double>(curr_pix.R) + static_cast<double>(curr_pix.G) + static_cast<double>(curr_pix.B)) / 3);
                if (!put_pixel(x, y, mean_color, mean_color, mean_color))
                {
                    cout << "Unexpected error occured while tried to turn picture into black-and-white format." << endl;
                    return 0;
                }
            }
        }
        return 1;
    }
    int cut(int x_margin, int y_margin)
    {
        if (x_margin < 0 || y_margin < 0 || (x_margin == 0 && y_margin == 0))
        {
            cout << "Error: incorrect margin parameters passed." << endl;
            return 0;
        }
        if (2 * x_margin >= width || 2 * y_margin >= height)
        {
            cout << "Error: incorrect margin parameters passed." << endl;
            return 0;
        }
        Canvas result(width - 2 * x_margin, height - 2 * y_margin);
        for (int x = x_margin; x < width - x_margin; x++)
        {
            for (int y = y_margin; y < height - y_margin; y++)
            {
                Pixel tmp = get_pixel(x, y);
                if (!result.put_pixel(x - x_margin, y - y_margin, tmp.R, tmp.G, tmp.B))
                {
                    cout << "Error occured while tried to cut canvas." << endl;
                    return 0;
                }
            }
        }
        this->width = result.width;
        this->height = result.height;
        this->image = result.image;
        return 1;
    }
    void help() const
    {
        cout << "Constructors: " << endl;
        cout << ">- canvas(int x, int y) - creates new canvas x * y. " << endl;
        cout << ">- canvas(string filename) - creates new canvas by loading *.bmp image from `filename`. " << endl << endl;

        cout << "Methods: " << endl;
        cout << ">- canvas.save(string filename) - saves canvas as *.bmp file `filename`. " << endl;
        cout << ">- canvas.put_pixel(int x, int y, int R, int G, int B) - draws pixel (color = RGB) at position (x,y) from upper left corner. " << endl;
        cout << ">- canvas.get_pixel(int x, int y) - returns Pixel object which contains x,y - positon and R,G,B - color. " << endl;
        cout << ">- canvas.fill_canvas(int R, int G, int B) - fills all canvas with R,G,B color. " << endl;
        cout << ">- canvas.fill_area(int x, int y, int R, int G, int B, int R_stop = -1, int G_stop = -1, int B_stop = -1)" << endl;
        cout << "   - fills area that conatins given dot with given R,G,B color. Passing default arguements R_stop, G_stop, B_stop " << endl;
        cout << "   will fill area inside of border with given color" << endl;
        cout << ">- canvas.draw_segment(int x1, int y1, int x2, int y2, int R, int G, int B) - draws a straight line from (x1, y1) to (x2, y2). " << endl;
        cout << ">- canvas.draw_segment(int x1, int y1, int x2, int y2, int w, int R, int G, int B) - draws a straight line from (x1, y1) to (x2, y2) with width = w." << endl;
        cout << ">- canvas.draw_rectangle(int x1, int y1, int x2, int y2, int R, int G, int B, int w = 1, int R_f, int G_f, int B_f) - " << endl;
        cout << "   - draws a rectangle with sides parallel to axis X and Y, which diagonal is segment (x1,y1)--(x2,y2). w - width of rectangle`s sides, " << endl;
        cout << "   R_f, G_f, B_f - fill color. (no filling if passed as default)." << endl;
        cout << ">- canvas.draw_triangle(int x1, int y1, int x2, int y2, int x3, int y3, int R, int G, int B, int R_f, int G_f, int B_f) - " << endl;
        cout << "   - draws a triangle with verticies (x1,y1), (x2,y2), (x3,y3) and border color R,G,B. If passed fill color, trinagle will be filled with " << endl;
        cout << "   R_f, G_f, B_f color ignoring all inner constructions. " << endl;
        cout << ">- canvas.draw_circle(int x, int y, int Rad, int R, int G, int B, int R_f, int G_f, int B_f) - draws a circle with center " << endl;
        cout << "   x,y and radius Rad, color = RGB, fill color = R_f, G_f, B_f" << endl;
        cout << ">- canvas.draw_circle(int x, int y, int Rad, int R, int G, int B, int w, int R_f, int G_f, int B_f) - all the same but you can pass " << endl;
        cout << "   a circle width w. " << endl;
        cout << ">- canvas.plot(double* var_values, double* func_values, int sample_count) - draws a plot, where var_values is an array " << endl;
        cout << "   with variable values and func_values is an array with corresponding function values. Sample_count is number of dots ( = length of arrays). " << endl;
        cout << ">- canvas.scatterplot(double* var_values, double* func_values, int sample_counts) - works the same way as canvas.plot, draws a scatter plot " << endl;
        cout << "   from values in var_values and corresponding function values in func_values. " << endl;
        cout << ">- canvas.copy_fragment(int x1, int y1, int x2, int y2) - returns canvas which is rectangular fragment of original from x1,y1 to x2,y2" << endl;
        cout << ">- canvas.insert_fragment(int x, int y, Canvas fragment) - inserts a fragment in current canvas, x,y is upper left point of fragment start" << endl;
        cout << ">- canvas.insert_fragment(int x, int y, string filename) - -//-, but fragment is taken from file `filename`. " << endl;
    }
};

#endif//CANVAS