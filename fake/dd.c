//计算算法中的Fh Fv
void CHandWritingNorm::ComputeFHV( IplImage* src,  IplImage* dist )
{
    CvSize imgsize = cvSize(src->width,src->height);
    cvZero(dist);
    unsigned char* data_ptr = (unsigned char*)src->imageData;
    float*         dist_ptr = (float *)dist->imageData;
    unsigned srcstep = src->widthStep;
    unsigned diststep = dist->widthStep/sizeof(float);
    //---------------------------------------------------------------
//begin scan line full fill Fh Fv
    for( int j = 0; j < imgsize.height; j++)
    {
        int last_pos = 0;
        for( int i = 0; i < imgsize.width; )
        {
            //-----------------------------------------------------
//inside one line
            for(; i < imgsize.width && IS_COMPONENT_POINT(data_ptr[i]); i++)
                ;//white back ground met black when out i is black,before i is white
            int length = i-last_pos+1;
            //length = std::min(1, std::max(1,length));
//--------------------------------------------------------
//fill fh,fv
            for( int tmpi = last_pos; tmpi < i; tmpi++)
                dist_ptr[tmpi] = 1.f/(float)length; //they are white ones
//set last_pos = i; i is black
            last_pos = i;
            if( i == imgsize.width)
                break;

            //---------------------------------------------------------
//find next white pixel
            for( ; i < imgsize.width && !IS_COMPONENT_POINT( data_ptr[i] ); i++)
                ;//i is white before i after last_pos is black;
            for( int tmpi = last_pos; tmpi < i; tmpi++)
                dist_ptr[tmpi] = 1.0f/(float)imgsize.width; //they are the black pixel
            last_pos = i;
        }//end process one line
//------------------------------------------------------
//move to next line
        data_ptr += srcstep;
        dist_ptr += diststep;
    }
}

//计算算法中的H V
void
CHandWritingNorm::ComputeHV()
{
    float* data_ptr = (float *)m_Fh->imageData;
    size_t step = m_Fh->widthStep/sizeof(float);
    float* data_ptr2 = (float *)m_Fv->imageData;
    size_t step2 = m_Fv->widthStep/sizeof(float);
    for(int i = 0; i < m_Fh->width; i++)
    {
        float hi = 0;
        float* datai_ptr = data_ptr;
        for( int j = 0; j < m_Fh->height; j++)
        {
            //CvScalar v = cvGet2D(m_Fh,j,i);
            hi += datai_ptr[i];
            datai_ptr += step;
        }
        m_H.push_back(hi);
    }//end compute H
    for(int i = 0; i < m_Fv->width; i++)
    {
        float vj = 0;
        float* dataj_ptr = data_ptr2;
        for( int j = 0; j < m_Fv->height; j++)
        {
            vj += dataj_ptr[i];
            dataj_ptr += step2;
        }
        m_V.push_back(vj);
    }//end compute V
}

//计算k,l的位置
void
CHandWritingNorm::ComputeKL( IplImage* dist )
{
    if( ( dist->nChannels!=1 ) || ( dist->depth != IPL_DEPTH_8U ) )
    {
        cout << "format not match" << endl;
        return;
    }
    cvZero(dist);
    //unsigned char *data_ptr = (unsigned char*)dist->imageData;
    CvSize imgsize = cvSize(dist->width,dist->height);
    //int A = imgsize.width/2;
//int B = imgsize.height/2;
    unsigned char* src_ptr = (unsigned char*)m_srcImage->imageData;
    //int lastk=0,lastl=0;
    for( int j = 0; j < m_srcImage->height; j++)
    {
        for( int i = 0; i < m_srcImage->width; i++)
        {
            if(!IS_COMPONENT_POINT(src_ptr[i]))
            {
                //calculate k l
                int k = (int)(0.1f*SumVector(m_H,i)+0.5f);
                int l = (int)(0.1f * SumVector(m_V,j)+0.5f);
                k = std::min(k,imgsize.width-1);
                l = std::min(l, imgsize.height-1);
                cvSet2D(dist,l,k,cvScalar(255));
            }
        }
        src_ptr += m_srcImage->widthStep;
    }
}