/* The MIT License

   Copyright (c) 2012-2015 Boston College.

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

/* The 2-clause BSD License

   Copyright 2006 Michael Farrar.  

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
   
   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
   
   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
 *  ssw.c
 *
 *  Created by Mengyao Zhao on 6/22/10.
 *  Copyright 2010 Boston College. All rights reserved.
 *	Version 1.2.4
 *	Last revision by Mengyao Zhao on 2019-03-04.
 *
 *  The lazy-F loop implementation was derived from SWPS3, which is
 *  MIT licensed under ETH Zürich, Institute of Computational Science.
 *
 *  The core SW loop referenced the swsse2 implementation, which is
 *  BSD licensed under Micharl Farrar.
 */

//#include <nmmintrin.h>
#include <emmintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ssw.h"

#ifdef __GNUC__
#define LIKELY(x) __builtin_expect((x),1)
#define UNLIKELY(x) __builtin_expect((x),0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

/* Convert the coordinate in the scoring matrix into the coordinate in one line of the band. */
#define set_u(u, w, i, j) { int x=(i)-(w); x=x>0?x:0; (u)=(j)-x+1; }

/* Convert the coordinate in the direction matrix into the coordinate in one line of the band. */
#define set_d(u, w, i, j, p) { int x=(i)-(w); x=x>0?x:0; x=(j)-x; (u)=x*3+p; }

/*! @function
  @abstract  Round an integer to the next closest power-2 integer.
  @param  x  integer to be rounded (in place)
  @discussion x will be modified.
 */
#define kroundup32(x) (--(x), (x)|=(x)>>1, (x)|=(x)>>2, (x)|=(x)>>4, (x)|=(x)>>8, (x)|=(x)>>16, ++(x))

typedef struct {
	uint16_t score;
	int32_t ref;	 //0-based position
	int32_t read;    //alignment ending position on read, 0-based
	// ADD
	uint16_t n_score;
	uint16_t* scoreArray;
	uint16_t* moveArray;
	int32_t refLen;
	int32_t readLen;
	//END
} alignment_end;

typedef struct {
	uint32_t* seq;
	int32_t length;
	int16_t read_begin;
	int16_t ref_begin;
	uint16_t insert_count;
	uint16_t delete_count;
	uint16_t match_count;
} cigar;

struct _profile{
	__m128i* profile_byte;	// 0: none
	__m128i* profile_word;	// 0: none
	const int16_t* read;
	const int8_t* mat;
	int32_t readLen;
	int32_t n;
	uint8_t bias;
};

/* array index is an ASCII character value from a CIGAR, 
   element value is the corresponding integer opcode between 0 and 8 */
const uint8_t encoded_ops[] = {
	0,         0,         0,         0,
	0,         0,         0,         0,
	0,         0,         0,         0,
	0,         0,         0,         0,
	0,         0,         0,         0,
	0,         0,         0,         0,
	0,         0,         0,         0,
	0,         0,         0,         0,
	0 /*   */, 0 /* ! */, 0 /* " */, 0 /* # */,
	0 /* $ */, 0 /* % */, 0 /* & */, 0 /* ' */,
	0 /* ( */, 0 /* ) */, 0 /* * */, 0 /* + */,
	0 /* , */, 0 /* - */, 0 /* . */, 0 /* / */,
	0 /* 0 */, 0 /* 1 */, 0 /* 2 */, 0 /* 3 */,
	0 /* 4 */, 0 /* 5 */, 0 /* 6 */, 0 /* 7 */,
	0 /* 8 */, 0 /* 9 */, 0 /* : */, 0 /* ; */,
	0 /* < */, 7 /* = */, 0 /* > */, 0 /* ? */,
	0 /* @ */, 0 /* A */, 0 /* B */, 0 /* C */,
	2 /* D */, 0 /* E */, 0 /* F */, 0 /* G */,
	5 /* H */, 1 /* I */, 0 /* J */, 0 /* K */,
	0 /* L */, 0 /* M */, 3 /* N */, 0 /* O */,
	6 /* P */, 0 /* Q */, 0 /* R */, 4 /* S */,
	0 /* T */, 0 /* U */, 0 /* V */, 0 /* W */,
	8 /* X */, 0 /* Y */, 0 /* Z */, 0 /* [ */,
	0 /* \ */, 0 /* ] */, 0 /* ^ */, 0 /* _ */,
	0 /* ` */, 0 /* a */, 0 /* b */, 0 /* c */,
	0 /* d */, 0 /* e */, 0 /* f */, 0 /* g */,
	0 /* h */, 0 /* i */, 0 /* j */, 0 /* k */,
	0 /* l */, 0 /* m */, 0 /* n */, 0 /* o */,
	0 /* p */, 0 /* q */, 0 /* r */, 0 /* s */,
	0 /* t */, 0 /* u */, 0 /* v */, 0 /* w */,
	0 /* x */, 0 /* y */, 0 /* z */, 0 /* { */,
	0 /* | */, 0 /* } */, 0 /* ~ */, 0 /*  */
};

/* Generate query profile rearrange query sequence & calculate the weight of match/mismatch. */
static __m128i* qP_byte (const int16_t* read_num,
				  const int8_t* mat,
				  const int32_t readLen,
				  const int32_t n,	/* the edge length of the squre matrix mat */
				  uint8_t bias) {

	int32_t segLen = (readLen + 15) / 16; /* Split the 128 bit register into 16 pieces.
								     Each piece is 8 bit. Split the read into 16 segments.
								     Calculat 16 segments in parallel.
								   */
	__m128i* vProfile = (__m128i*)malloc(n * segLen * sizeof(__m128i));
	int8_t* t = (int8_t*)vProfile;
	int32_t nt, i, j, segNum;

	/* Generate query profile rearrange query sequence & calculate the weight of match/mismatch */
	for (nt = 0; LIKELY(nt < n); nt ++) {
		for (i = 0; i < segLen; i ++) {
			j = i;
			for (segNum = 0; LIKELY(segNum < 16) ; segNum ++) {
				*t++ = j>= readLen ? bias : mat[nt * n + read_num[j]] + bias;
				j += segLen;
			}
		}
	}
	return vProfile;
}

/* Striped Smith-Waterman
   Record the highest score of each reference position.
   Return the alignment score and ending position of the best alignment, 2nd best alignment, etc.
   Gap begin and gap extension are different.
   wight_match > 0, all other weights < 0.
   The returned positions are 0-based.
 */
static alignment_end* sw_sse2_byte (const int16_t* ref,
							 int8_t ref_dir,	// 0: forward ref; 1: reverse ref
							 int32_t refLen,
							 int32_t readLen,
							 const uint8_t weight_gapO, /* will be used as - */
							 const uint8_t weight_gapE, /* will be used as - */
							 const __m128i* vProfile,
							 uint8_t terminate,	/* the best alignment score: used to terminate
												   the matrix calculation when locating the
												   alignment beginning point. If this score
												   is set to 0, it will not be used */
	 						 uint8_t bias,  /* Shift 0 point to a positive value. */
							 int32_t maskLen) {

// Put the largest number of the 16 numbers in vm into m.
#define max16(m, vm) (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 8)); \
					  (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 4)); \
					  (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 2)); \
					  (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 1)); \
					  (m) = _mm_extract_epi16((vm), 0)

	uint8_t max = 0;		                     /* the max alignment score */
	uint8_t n_max = 0;		                     /* the max alignment score */
	int32_t end_read = readLen - 1;
	int32_t end_ref = -1; /* 0_based best alignment ending point; Initialized as isn't aligned -1. */
	int32_t segLen = (readLen + 15) / 16; /* number of segment */

	/* array to record the largest score of each reference position */
	uint8_t* maxColumn = (uint8_t*) calloc(refLen, 1);

	/* array to record the alignment read ending position of the largest score of each reference position */
	int32_t* end_read_column = (int32_t*) calloc(refLen, sizeof(int32_t));

	/* Define 16 byte 0 vector. */
	__m128i vZero = _mm_set1_epi32(0);

	__m128i* pvHStore = (__m128i*) calloc(segLen, sizeof(__m128i));
	
		// [ ADD ] 
	__m128i* pvHALLStore = (__m128i*) calloc(refLen*segLen, sizeof(__m128i)); //newly add 2020.03.20 Joey
	__m128i* pvHStepStore = (__m128i*) calloc(refLen*segLen, sizeof(__m128i)); //newly add 2020.03.20 Joey
	// Step:  1: 斜, 2:水平, 3: 垂直
	__m128i  vOne=_mm_set1_epi8(1);
	__m128i  vMask2 = _mm_set1_epi8(2);
	__m128i  vMask4 = _mm_set1_epi8(4);
	// __m128i vFF = _mm_set1_epi8(0xff); 
	// end

	__m128i* pvHLoad = (__m128i*) calloc(segLen, sizeof(__m128i));
	__m128i* pvE = (__m128i*) calloc(segLen, sizeof(__m128i));
	__m128i* pvHmax = (__m128i*) calloc(segLen, sizeof(__m128i));

	int32_t i, j, k;
	/* 16 byte insertion begin vector */
	__m128i vGapO = _mm_set1_epi8(weight_gapO);

	/* 16 byte insertion extension vector */
	__m128i vGapE = _mm_set1_epi8(weight_gapE);

	/* 16 byte bias vector */
	__m128i vBias = _mm_set1_epi8(bias);

	__m128i vMaxScore = vZero; /* Trace the highest score of the whole SW matrix. */
	__m128i vMaxMark = vZero; /* Trace the highest score till the previous column. */
	__m128i vTemp;
	int32_t edge, begin = 0, end = refLen, step = 1;

	/* outer loop to process the reference sequence */
	if (ref_dir == 1) {
		begin = refLen - 1;
		end = -1;
		step = -1;
	}
	// 這裡i 與最後的maxColumn[i]的關係
	for (i = begin; LIKELY(i != end); i += step) {
		int32_t cmp;
		__m128i e, vF = vZero, vMaxColumn = vZero; /* Initialize F value to 0.
							   Any errors to vH values will be corrected in the Lazy_F loop.
							 */

		__m128i vH = pvHStore[segLen - 1];
		vH = _mm_slli_si128 (vH, 1); /* Shift the 128-bit value in vH left by 1 byte. */
		const __m128i* vP = vProfile + ref[i] * segLen; /* Right part of the vProfile */

		/* Swap the 2 H buffers. */
		__m128i* pv = pvHLoad;
		pvHLoad = pvHStore;
		pvHStore = pv;

		// [ADD] 2020.03.10 比較比對前後
		__m128i vX ;
		// end

		/* inner loop to process the query sequence */
		for (j = 0; LIKELY(j < segLen); ++j) {
			//由 vP+j 讀出來，加上 vH，放入 vH
			vH = _mm_adds_epu8(vH, _mm_load_si128(vP + j));
			// (vP + j) = 可以理解為: point(i,j)
			// vH : 迴圈最後，vH = _mm_load_si128(pvHLoad + j); 所以對於本圈，j 需 -1
			// 而 pvHLoad 是在全圈外指定為pvHSave，也就是 i-1 圈 的vH值
			// 所以 這裡的vH 可以理解為 range(i-1,j-1)
			// 所以這裡的 vH 的計算為 D => range(i-1,j-1) + point(i,j)
			
			// [ADD] 2020.03.10 增
			__m128i vMove = vOne;
			_mm_store_si128(pvHStepStore+i*segLen+j, vOne);
			//end

			vH = _mm_subs_epu8(vH, vBias); /* vH will be always > 0 */

			/* Get max from vH, vE and vF. */
			e = _mm_load_si128(pvE + j);
			// 這裡的 pvE +j 是在下面的save
			// e = _mm_max_epu8(e, vH);
			// _mm_store_si128(pvE + j, e);
			// 所以是前i輪 j 的 vH - GAP
			// 可以理解為 U(=i的變化), => range(i-1,j) -GAP  

			// [ADD] 2020.03.10 增
				vX=vH;  //備份
			// end

			vH = _mm_max_epu8(vH, e);
			// 決定是 vH 與 e的大小

			// [ADD] 2020.03.10 增
			// 找出變動部份
			vTemp = ~_mm_cmpeq_epi8(vX, vH);
			// 製作更新
			vTemp = _mm_and_si128(vTemp,vMask2);
			// 更新到 vMove
			vMove = _mm_max_epu8(vMove, vTemp);// end

			// [ADD] 2020.03.10 增
				vX=vH;  //備份
			// end

			vH = _mm_max_epu8(vH, vF);
			// VF 根本還沒算，是因為VF是上一輪j 的計算結果
			// 因為 j,segLen 的擺法，會變成： (0,2,4,6,8)  (1,3,5,7,9)  這樣
			// 所以上一 j 將會是 前一位置。  
			// 上一j 計算：	
			// vF = _mm_subs_epu8(vF, vGapE);
			// vF = _mm_max_epu8(vF, vH);
			// 所以這裡的 vF 可以理解為： L(=j的變化) =>  range(i,j-1) - GapE
			
			// [ADD] 2020.03.10 增
			vTemp = ~_mm_cmpeq_epi8(vX, vH);
			vTemp = _mm_and_si128(vTemp,vMask4);
			vMove = _mm_max_epu8(vMove, vTemp);

			//儲存vMove
			_mm_store_si128(pvHStepStore+i*segLen+j, vMove);
			// end

			vMaxColumn = _mm_max_epu8(vMaxColumn, vH);
				// 經過以上兩比較，分數計算完成

			/* Save vH values. */
			_mm_store_si128(pvHStore + j, vH);

			/* Update vE value. */
			vH = _mm_subs_epu8(vH, vGapO); /* saturation arithmetic, result >= 0 */
			e = _mm_subs_epu8(e, vGapE);
			e = _mm_max_epu8(e, vH);
			_mm_store_si128(pvE + j, e);

			/* Update vF value. */
			// vF 第一輪 為0. 第2輪這段 會變成 vH - Gap 
			vF = _mm_subs_epu8(vF, vGapE);
			// vF 第一輪 這段會變 vH
			vF = _mm_max_epu8(vF, vH);

			/* Load the next vH. */
			vH = _mm_load_si128(pvHLoad + j);
		}

/* Lazy_F loop: has been revised to disallow adjecent insertion and then deletion, so don't update E(i, j), learn from SWPS3 */
		for (k = 0; LIKELY(k < 16); ++k) {
			vF = _mm_slli_si128 (vF, 1);
			for (j = 0; LIKELY(j < segLen); ++j) {
				//由pvHStore 讀入有關 區塊 j 的暫存
				vH = _mm_load_si128(pvHStore + j);
				// Guess：vH, vF (2), 比的前後看看是 VH 還是 VF

				// [ADD] 2020/03/11 DEBUG
				vX=vH;  //備份
				//END

				vH = _mm_max_epu8(vH, vF);
				// 以上。vH 完成所有運算

				// [ADD] 2020/03/11 DEBUG
				vTemp = _mm_cmpeq_epi8(vX, vH);
				cmp = _mm_movemask_epi8(vTemp);
				// if (cmp != 0xffff) {
				// 	printf("Change:{%d,%d}\n",i,j);
				// }
				//END
				vTemp = ~_mm_cmpeq_epi8(vX, vH);
				vTemp = _mm_and_si128(vTemp,vMask4);
				__m128i vMove  = _mm_load_si128(pvHStepStore+i*segLen+j);
				vMove = _mm_max_epu8(vMove, vTemp);
				_mm_store_si128(pvHStepStore+i*segLen+j, vMove);


				vMaxColumn = _mm_max_epu8(vMaxColumn, vH);	// newly added line
				_mm_store_si128(pvHStore + j, vH);

				vH = _mm_subs_epu8(vH, vGapO);
				vF = _mm_subs_epu8(vF, vGapE);
				if (UNLIKELY(! _mm_movemask_epi8(_mm_cmpgt_epi8(vF, vH)))) goto end;
			}
		}

end:		
		// 簡單認為到這邊 [i] 的所有比對已經完成，已經走完所有的[j]
		// 把vMaxColumn 與之前最好的vMaxScore, 整合出max答案。
		vMaxScore = _mm_max_epu8(vMaxScore, vMaxColumn);
		// 比對 vMaxMark, vMaxScore，1 byte 比，有16組答案，若相同回應 0xff, 不相同 =0
		vTemp = _mm_cmpeq_epi8(vMaxMark, vMaxScore);
		//抓最高位的bit, 組合成一個數
		cmp = _mm_movemask_epi8(vTemp);
		//如果不是所有數字都相等, 那就可能有最大數字
		if (cmp != 0xffff) {
			uint8_t temp;
			vMaxMark = vMaxScore;
			//找出最大數字，不管它在哪，放到 temp 去(單一整數)
			max16(temp, vMaxScore);
			vMaxScore = vMaxMark;

			// 新的最大數字大於 原有max, 進行更新
			if (LIKELY(temp > max)) {
				max = temp;
				if (max + bias >= 255) break;	//overflow
				end_ref = i;

				// 所以 pvHmax[j] 就是最大的那一串
				/* Store the column with the highest alignment score in order to trace the alignment ending position on read. */
				for (j = 0; LIKELY(j < segLen); ++j) pvHmax[j] = pvHStore[j];
			}	
		}

		//[ADD ]試著去儲存所有分數 2020/03/10
		for (j = 0; LIKELY(j < segLen); ++j) {
				//_mm_store_si128(pvHALLStore+i*segLen+j, pvHStore[j]);
				pvHALLStore[i*segLen+j] = pvHStore[j];
		}

		/* Record the max score of current column. */
		max16(maxColumn[i], vMaxColumn);
		if (maxColumn[i] == terminate) break;
	} // end of [i] loop

	/* Trace the alignment ending position on read. */
	uint8_t *t = (uint8_t*)pvHmax;
	int32_t column_len = segLen * 16;
	for (i = 0; LIKELY(i < column_len); ++i, ++t) {
		int32_t temp;
		if (*t == max) {
			temp = i / 16 + i % 16 * segLen;
			if (temp < end_read) end_read = temp;
		}
	}

	uint16_t *score_16=0, *move_16 = 0;

		//  [ADD]  2020/03/10 Try to Dump pvHALLStore, pvHStepStore
	if (ref_dir==0){
		uint8_t *score = (uint8_t*)pvHALLStore;
		uint8_t *move = (uint8_t*)pvHStepStore; //move
	
		score_16 = (uint16_t*)malloc( refLen* readLen * sizeof(uint16_t));
		move_16 = (uint16_t*)malloc( refLen* readLen * sizeof(uint16_t));

		//製做i,j 對應的score, step Array

		for (i=0; i<refLen; ++i){
			for (j = 0; LIKELY(j < readLen); ++j) {
				// 計算第i,j值，需要讀取哪個位置
				int readpos = i*column_len+(j % segLen)*16+  j/segLen;
				int writepos = i*readLen+j;
				score_16[writepos]=(uint16_t)score[readpos];
				move_16[writepos]=(uint16_t)move[readpos];
				}
		}
		n_max = score_16[end_ref*readLen+end_read];
		// printf("最佳分數 (%d,%d):%d\n",end_ref,end_read,n_max);

	// 	for (i=0; i<refLen; ++i){
	// 		for (j = 0; LIKELY(j < readLen ); ++j) {
	// 			// 計算第i,j值，需要讀取哪個位置
	// 			printf("[%d,%d]%d(%d) ",i,j,score_16[i*readLen+j],move_16[i*readLen+j]);
	// 		}
	// 		printf("\n");
	// 	}
	}
	//end

	/* Find the most possible 2nd best alignment. */
	alignment_end* bests = (alignment_end*) calloc(2, sizeof(alignment_end));
	bests[0].score = max + bias >= 255 ? 255 : max;
	bests[0].ref = end_ref;
	bests[0].read = end_read;

	// [ADD] 2020.03.11
	bests[0].n_score = n_max;
	bests[0].scoreArray =score_16;
	bests[0].moveArray=move_16;
	bests[0].refLen = refLen;
	bests[0].readLen= readLen;
	// end

	bests[1].score = 0;
	bests[1].ref = 0;
	bests[1].read = 0;

	//遮住 (end_ref-mask_Len) --> (end_ref+mask_Len)
	//然後在其他的Regin, 用maxColumn[i] 找 second best

	edge = (end_ref - maskLen) > 0 ? (end_ref - maskLen) : 0;
	for (i = 0; i < edge; i ++) {
		if (maxColumn[i] > bests[1].score) {
			bests[1].score = maxColumn[i];
			bests[1].ref = i;
		}
	}
	edge = (end_ref + maskLen) > refLen ? refLen : (end_ref + maskLen);
	for (i = edge + 1; i < refLen; i ++) {
		if (maxColumn[i] > bests[1].score) {
			bests[1].score = maxColumn[i];
			bests[1].ref = i;
		}
	}

	free(pvHmax);
	free(pvHALLStore); // [ADD] 2020/03/10
	free(pvHStepStore); // [ADD] 2020/03/10
	free(pvE);
	free(pvHLoad);
	free(pvHStore);

	free(maxColumn);
	free(end_read_column);
	return bests;
}

static __m128i* qP_word (const int16_t* read_num,
				  const int8_t* mat,
				  const int32_t readLen,
				  const int32_t n) {

	int32_t segLen = (readLen + 7) / 8;
	__m128i* vProfile = (__m128i*)malloc(n * segLen * sizeof(__m128i));
	int16_t* t = (int16_t*)vProfile;
	int32_t nt, i, j;
	int32_t segNum;

	/* Generate query profile rearrange query sequence & calculate the weight of match/mismatch */
	for (nt = 0; LIKELY(nt < n); nt ++) {
		for (i = 0; i < segLen; i ++) {
			j = i;
			for (segNum = 0; LIKELY(segNum < 8) ; segNum ++) {
				*t++ = j>= readLen ? 0 : mat[nt * n + read_num[j]];
				j += segLen;
			}
		}
	}
	return vProfile;
}

static alignment_end* sw_sse2_word (const int16_t* ref,
							 int8_t ref_dir,	// 0: forward ref; 1: reverse ref
							 int32_t refLen,
							 int32_t readLen,
							 const uint8_t weight_gapO, /* will be used as - */
							 const uint8_t weight_gapE, /* will be used as - */
							 const __m128i* vProfile,
							 uint16_t terminate,
							 int32_t maskLen) {

#define max8(m, vm) (vm) = _mm_max_epi16((vm), _mm_srli_si128((vm), 8)); \
					(vm) = _mm_max_epi16((vm), _mm_srli_si128((vm), 4)); \
					(vm) = _mm_max_epi16((vm), _mm_srli_si128((vm), 2)); \
					(m) = _mm_extract_epi16((vm), 0)

	uint16_t max = 0;		                     /* the max alignment score */
	uint16_t n_max = 0;	
	int32_t end_read = readLen - 1;
	int32_t end_ref = 0; /* 1_based best alignment ending point; Initialized as isn't aligned - 0. */
	int32_t segLen = (readLen + 7) / 8; /* number of segment */

	/* array to record the largest score of each reference position */
	uint16_t* maxColumn = (uint16_t*) calloc(refLen, 2);

	/* array to record the alignment read ending position of the largest score of each reference position */
	int32_t* end_read_column = (int32_t*) calloc(refLen, sizeof(int32_t));

	/* Define 16 byte 0 vector. */
	__m128i vZero = _mm_set1_epi32(0);

	__m128i* pvHStore = (__m128i*) calloc(segLen, sizeof(__m128i));
	
		// [ ADD ] 
	__m128i* pvHALLStore = (__m128i*) calloc(refLen*segLen, sizeof(__m128i)); //newly add 2020.03.20 Joey
	__m128i* pvHStepStore = (__m128i*) calloc(refLen*segLen, sizeof(__m128i)); //newly add 2020.03.20 Joey
	// Step:  1: 斜, 2:水平, 3: 垂直
	__m128i  vOne=_mm_set1_epi16(1);
	__m128i  vMask2 = _mm_set1_epi16(2);
	__m128i  vMask4 = _mm_set1_epi16(4);
	// __m128i vFF = _mm_set1_epi16(0xffff); 
	// end

	__m128i* pvHLoad = (__m128i*) calloc(segLen, sizeof(__m128i));
	__m128i* pvE = (__m128i*) calloc(segLen, sizeof(__m128i));
	__m128i* pvHmax = (__m128i*) calloc(segLen, sizeof(__m128i));

	int32_t i, j, k;
	/* 16 byte insertion begin vector */
	__m128i vGapO = _mm_set1_epi16(weight_gapO);

	/* 16 byte insertion extension vector */
	__m128i vGapE = _mm_set1_epi16(weight_gapE);

	__m128i vMaxScore = vZero; /* Trace the highest score of the whole SW matrix. */
	__m128i vMaxMark = vZero; /* Trace the highest score till the previous column. */
	__m128i vTemp;
	int32_t edge, begin = 0, end = refLen, step = 1;

	/* outer loop to process the reference sequence */
	if (ref_dir == 1) {
		begin = refLen - 1;
		end = -1;
		step = -1;
	}
	for (i = begin; LIKELY(i != end); i += step) {
		int32_t cmp;
		__m128i e, vF = vZero; /* Initialize F value to 0.
							   Any errors to vH values will be corrected in the Lazy_F loop.
							 */
		__m128i vH = pvHStore[segLen - 1];
		vH = _mm_slli_si128 (vH, 2); /* Shift the 128-bit value in vH left by 2 byte. */

		/* Swap the 2 H buffers. */
		__m128i* pv = pvHLoad;

		__m128i vMaxColumn = vZero; /* vMaxColumn is used to record the max values of column i. */

		const __m128i* vP = vProfile + ref[i] * segLen; /* Right part of the vProfile */
		pvHLoad = pvHStore;
		pvHStore = pv;

		// [ADD] 2020.03.10 比較比對前後
		__m128i vX, vMove ;
		// end

		/* inner loop to process the query sequence */
		for (j = 0; LIKELY(j < segLen); j ++) {
			vH = _mm_adds_epi16(vH, _mm_load_si128(vP + j));

			vMove=vOne;  // First Move

			/* Get max from vH, vE and vF. */
			e = _mm_load_si128(pvE + j);

			vX=vH;  //備份

			vH = _mm_max_epi16(vH, e); // 第一次更新 vH

			vTemp = ~_mm_cmpeq_epi16(vX, vH); // 比對前後
			vTemp = _mm_and_si128(vTemp,vMask2); // 找出變化
			vMove = _mm_max_epi16(vMove, vTemp);  //更新vMove

			vX=vH;  //備份

			vH = _mm_max_epi16(vH, vF);
			vMaxColumn = _mm_max_epi16(vMaxColumn, vH);

			vTemp = ~_mm_cmpeq_epi16(vX, vH); // 比對前後
			vTemp = _mm_and_si128(vTemp,vMask4); // 找出變化
			vMove = _mm_max_epi16(vMove, vTemp);  //更新vMove

			//儲存vMove
			_mm_store_si128(pvHStepStore+i*segLen+j, vMove);

			/* Save vH values. */
			_mm_store_si128(pvHStore + j, vH);

			/* Update vE value. */
			vH = _mm_subs_epu16(vH, vGapO); /* saturation arithmetic, result >= 0 */
			e = _mm_subs_epu16(e, vGapE);
			e = _mm_max_epi16(e, vH);
			_mm_store_si128(pvE + j, e);

			/* Update vF value. */
			vF = _mm_subs_epu16(vF, vGapE);
			vF = _mm_max_epi16(vF, vH);

			/* Load the next vH. */
			vH = _mm_load_si128(pvHLoad + j);
		}

		/* Lazy_F loop: has been revised to disallow adjecent insertion and then deletion, so don't update E(i, j), learn from SWPS3 */
		for (k = 0; LIKELY(k < 8); ++k) {
			vF = _mm_slli_si128 (vF, 2);
			for (j = 0; LIKELY(j < segLen); ++j) {
				vH = _mm_load_si128(pvHStore + j);
				
				vX=vH;  //備份

				vH = _mm_max_epi16(vH, vF);

				vTemp = ~_mm_cmpeq_epi16(vX, vH); //比對前後
				vTemp = _mm_and_si128(vTemp,vMask4); //找出變化
				__m128i vMove  = _mm_load_si128(pvHStepStore+i*segLen+j);
				vMove = _mm_max_epi16(vMove, vTemp); //更新 vMove
				_mm_store_si128(pvHStepStore+i*segLen+j, vMove); //儲存vMove

				vMaxColumn = _mm_max_epi16(vMaxColumn, vH); //newly added line
				_mm_store_si128(pvHStore + j, vH);
				vH = _mm_subs_epu16(vH, vGapO);
				vF = _mm_subs_epu16(vF, vGapE);
				if (UNLIKELY(! _mm_movemask_epi8(_mm_cmpgt_epi16(vF, vH)))) goto end;
			}
		}

end:
		vMaxScore = _mm_max_epi16(vMaxScore, vMaxColumn);
		vTemp = _mm_cmpeq_epi16(vMaxMark, vMaxScore);
		cmp = _mm_movemask_epi8(vTemp);
		if (cmp != 0xffff) {
			uint16_t temp;
			vMaxMark = vMaxScore;
			max8(temp, vMaxScore);
			vMaxScore = vMaxMark;

			if (LIKELY(temp > max)) {
				max = temp;
				end_ref = i;
				for (j = 0; LIKELY(j < segLen); ++j) pvHmax[j] = pvHStore[j];
			}
		}

		//[ADD ]試著去儲存所有分數 2020/03/10
		for (j = 0; LIKELY(j < segLen); ++j) {
				//_mm_store_si128(pvHALLStore+i*segLen+j, pvHStore[j]);
				pvHALLStore[i*segLen+j] = pvHStore[j];
		}

		/* Record the max score of current column. */
		max8(maxColumn[i], vMaxColumn);
		if (maxColumn[i] == terminate) break;
	}

	/* Trace the alignment ending position on read. */
	uint16_t *t = (uint16_t*)pvHmax;
	int32_t column_len = segLen * 8;
	for (i = 0; LIKELY(i < column_len); ++i, ++t) {
		int32_t temp;
		if (*t == max) {
			temp = i / 8 + i % 8 * segLen;
			if (temp < end_read) end_read = temp;
		}
	}

	uint16_t *score_16 = 0, *move_16 = 0;

	//  [ADD]  2020/03/10 Try to Dump pvHALLStore, pvHStepStore
	if (ref_dir==0){
		uint16_t *score = (uint16_t*)pvHALLStore;
		uint16_t *move = (uint16_t*)pvHStepStore; //move

		score_16 = (uint16_t*)malloc( refLen* readLen * sizeof(uint16_t));
		move_16 = (uint16_t*)malloc( refLen* readLen* sizeof(uint16_t));

	//製做i,j 對應的score, step Array

		for (i=0; i<refLen; ++i){
			for (j = 0; LIKELY(j < readLen); ++j) {
				// 計算第i,j值，需要讀取哪個位置
				int readpos = i*column_len+(j % segLen)*8+  j/segLen;
				int writepos = i*readLen+j;
				score_16[writepos]=(uint16_t)score[readpos];
				move_16[writepos]=(uint16_t)move[readpos];
				}
		}
		n_max = score_16[end_ref*readLen+end_read];
		// printf("最佳分數 (%d,%d):%d\n",end_ref,end_read,n_max);

		// for (i=0; i<refLen; ++i){
		// 	for (j = 0; LIKELY(j < readLen ); ++j) {
		// 		// 計算第i,j值，需要讀取哪個位置
		// 		printf("[%d,%d]%d(%d) ",i,j,score_16[i*readLen+j],move_16[i*readLen+j]);
		// 	}
		// 	printf("\n");
		// }

	 }
	//end

	/* Find the most possible 2nd best alignment. */
	alignment_end* bests = (alignment_end*) calloc(2, sizeof(alignment_end));
	bests[0].score = max;
	bests[0].ref = end_ref;
	bests[0].read = end_read;

// [ADD] 2020.03.11
	// 設定 scoreAarry 的型態 (1: uint8_t, 2: uint16_t)
	bests[0].n_score = n_max;
	bests[0].refLen = refLen;
	bests[0].readLen= readLen;

	bests[0].scoreArray = score_16;
	bests[0].moveArray= move_16;

	bests[1].score = 0;
	bests[1].ref = 0;
	bests[1].read = 0;

	edge = (end_ref - maskLen) > 0 ? (end_ref - maskLen) : 0;
	for (i = 0; i < edge; i ++) {
		if (maxColumn[i] > bests[1].score) {
			bests[1].score = maxColumn[i];
			bests[1].ref = i;
		}
	}
	edge = (end_ref + maskLen) > refLen ? refLen : (end_ref + maskLen);
	for (i = edge; i < refLen; i ++) {
		if (maxColumn[i] > bests[1].score) {
			bests[1].score = maxColumn[i];
			bests[1].ref = i;
		}
	}

	free(pvHmax);
	free(pvHALLStore); // [ADD] 2020/03/10
	free(pvHStepStore); // [ADD] 2020/03/10
	free(pvE);
	free(pvHLoad);
	free(pvHStore);

	free(maxColumn);
	free(end_read_column);
	return bests;
}

static cigar* banded_sw (const int16_t* ref,
				 const int16_t* read,
				 int32_t refLen,
				 int32_t readLen,
				 int32_t score,
				 const uint32_t weight_gapO,  /* will be used as - */
				 const uint32_t weight_gapE,  /* will be used as - */
				 int32_t band_width,
				 const int8_t* mat,	/* pointer to the weight matrix */
				 int32_t n) {

	uint32_t *c = (uint32_t*)malloc(16 * sizeof(uint32_t)), *c1;
	int32_t i, j, e, f, temp1, temp2, s = 16, s1 = 8, l, max = 0;
	int64_t s2 = 1024;
	char op, prev_op;
	int32_t width, width_d, *h_b, *e_b, *h_c;
	int8_t *direction, *direction_line;
	cigar* result = (cigar*)malloc(sizeof(cigar));
	h_b = (int32_t*)malloc(s1 * sizeof(int32_t));
	e_b = (int32_t*)malloc(s1 * sizeof(int32_t));
	h_c = (int32_t*)malloc(s1 * sizeof(int32_t));
	direction = (int8_t*)malloc(s2 * sizeof(int8_t));

	do {
		width = band_width * 2 + 3, width_d = band_width * 2 + 1;
		while (width >= s1) {
			++s1;
			kroundup32(s1);
			h_b = (int32_t*)realloc(h_b, s1 * sizeof(int32_t));
			e_b = (int32_t*)realloc(e_b, s1 * sizeof(int32_t));
			h_c = (int32_t*)realloc(h_c, s1 * sizeof(int32_t));
		}
		while (width_d * readLen * 3 >= s2) {
			++s2;
			kroundup32(s2);
			if (s2 < 0) {
				fprintf(stderr, "Alignment score and position are not consensus.\n");
				exit(1);
			}
			direction = (int8_t*)realloc(direction, s2 * sizeof(int8_t));
		}
		direction_line = direction;
		for (j = 1; LIKELY(j < width - 1); j ++) h_b[j] = 0;
		for (i = 0; LIKELY(i < readLen); i ++) {
			int32_t beg = 0, end = refLen - 1, u = 0, edge;
			j = i - band_width;	beg = beg > j ? beg : j; // band start
			j = i + band_width; end = end < j ? end : j; // band end
			edge = end + 1 < width - 1 ? end + 1 : width - 1;
			f = h_b[0] = e_b[0] = h_b[edge] = e_b[edge] = h_c[0] = 0;
			direction_line = direction + width_d * i * 3;

			for (j = beg; LIKELY(j <= end); j ++) {
				int32_t b, e1, f1, d, de, df, dh;
				set_u(u, band_width, i, j);	set_u(e, band_width, i - 1, j);
				set_u(b, band_width, i, j - 1); set_u(d, band_width, i - 1, j - 1);
				set_d(de, band_width, i, j, 0);
				set_d(df, band_width, i, j, 1);
				set_d(dh, band_width, i, j, 2);

				temp1 = i == 0 ? -weight_gapO : h_b[e] - weight_gapO;
				temp2 = i == 0 ? -weight_gapE : e_b[e] - weight_gapE;
				e_b[u] = temp1 > temp2 ? temp1 : temp2;
				direction_line[de] = temp1 > temp2 ? 3 : 2;

				temp1 = h_c[b] - weight_gapO;
				temp2 = f - weight_gapE;
				f = temp1 > temp2 ? temp1 : temp2;
				direction_line[df] = temp1 > temp2 ? 5 : 4;

				e1 = e_b[u] > 0 ? e_b[u] : 0;
				f1 = f > 0 ? f : 0;
				temp1 = e1 > f1 ? e1 : f1;
				temp2 = h_b[d] + mat[ref[j] * n + read[i]];
				h_c[u] = temp1 > temp2 ? temp1 : temp2;

				if (h_c[u] > max) max = h_c[u];

				if (temp1 <= temp2) direction_line[dh] = 1;
				else direction_line[dh] = e1 > f1 ? direction_line[de] : direction_line[df];
			}
			for (j = 1; j <= u; j ++) h_b[j] = h_c[j];
		}
		band_width *= 2;
	} while (LIKELY(max < score));
	band_width /= 2;

	// M, match , (i-1,j-1) -> (i, j)
	// I, Insert, (i-1,j) -> (i, j)
	// D, Delete  (i,j-1) -> (i, j)

	//這裡開始做 cigar, 
	
	// trace back
	i = readLen - 1;
	j = refLen - 1;
	e = 0;	// Count the number of M, D or I.
	l = 0;	// record length of current cigar
	op = prev_op = 'M';
	temp2 = 2;	// h
	while (LIKELY(i > 0) || LIKELY(j > 0)) {
		set_d(temp1, band_width, i, j, temp2);
		switch (direction_line[temp1]) {
			case 1:
				--i;
				--j;
				temp2 = 2;
				direction_line -= width_d * 3;
				op = 'M';
				break;
			case 2:
			 	--i;
				temp2 = 0;	// e
				direction_line -= width_d * 3;
				op = 'I';
				break;
			case 3:
				--i;
				temp2 = 2;
				direction_line -= width_d * 3;
				op = 'I';
				break;
			case 4:
				--j;
				temp2 = 1;
				op = 'D';
				break;
			case 5:
				--j;
				temp2 = 2;
				op = 'D';
				break;
			default:
				fprintf(stderr, "Trace back error: %d.\n", direction_line[temp1 - 1]);
				free(direction);
				free(h_c);
				free(e_b);
				free(h_b);
				free(c);
				free(result);
				return 0;
		}
		// e 是累進記數器
		if (op == prev_op) ++e;
		else {
			++l;  // 最後cigar 儲存位置
			while (l >= s) {  // 補記憶體
				++s;
				kroundup32(s);
				c = (uint32_t*)realloc(c, s * sizeof(uint32_t));
			}
			// 把長度與op, 壓縮入c
			c[l - 1] = to_cigar_int(e, prev_op);
			prev_op = op;
			e = 1;
		}
	}
	if (op == 'M') {
		++l;
		while (l >= s) {
			++s;
			kroundup32(s);
			c = (uint32_t*)realloc(c, s * sizeof(uint32_t));
		}
		c[l - 1] = to_cigar_int(e + 1, op);
	}else {
		l += 2;
		while (l >= s) {
			++s;
			kroundup32(s);
			c = (uint32_t*)realloc(c, s * sizeof(uint32_t));
		}
		c[l - 2] = to_cigar_int(e, op);
		c[l - 1] = to_cigar_int(1, 'M');
	}

	// reverse cigar
	c1 = (uint32_t*)malloc(l * sizeof(uint32_t));
	s = 0;
	e = l - 1;
	while (LIKELY(s <= e)) {
		c1[s] = c[e];
		c1[e] = c[s];
		++ s;
		-- e;
	}
	result->seq = c1;
	result->length = l;

	free(direction);
	free(h_c);
	free(e_b);
	free(h_b);
	free(c);
	return result;
}

static int16_t* seq_reverse(const int16_t* seq, int32_t end)	/* end is 0-based alignment ending position */
{
	int16_t* reverse = (int16_t*)calloc(end + 1, sizeof(int16_t));
	int32_t start = 0;
	while (LIKELY(start <= end)) {
		reverse[start] = seq[end];
		reverse[end] = seq[start];
		++ start;
		-- end;
	}
	return reverse;
}

s_profile* ssw_init (const int16_t* read, const int32_t readLen, const int8_t* mat, const int32_t n, const int8_t score_size) {
	s_profile* p = (s_profile*)calloc(1, sizeof(struct _profile));
	p->profile_byte = 0;
	p->profile_word = 0;
	p->bias = 0;

	if (score_size == 0 || score_size == 2) {
		/* Find the bias to use in the substitution matrix */
		int32_t bias = 0, i;
		for (i = 0; i < n*n; i++) if (mat[i] < bias) bias = mat[i];
		bias = abs(bias);

		p->bias = bias;
		p->profile_byte = qP_byte (read, mat, readLen, n, bias);
	}
	if (score_size == 1 || score_size == 2) p->profile_word = qP_word (read, mat, readLen, n);
	p->read = read;
	p->mat = mat;
	p->readLen = readLen;
	p->n = n;
	return p;
}

void init_destroy (s_profile* p) {
	free(p->profile_byte);
	free(p->profile_word);
	free(p);
}

cigar* back_trace(const alignment_end a){
	//回傳用的容器
	cigar* result = (cigar*)malloc(sizeof(cigar)); 
	//回傳前的準備
	uint32_t *c = (uint32_t*)malloc(16 * sizeof(uint32_t));
	uint32_t *c1 =0;

	// int16_t read_begin =-1;
	// int16_t ref_begin=-1;
	// int16_t insert_count=0;
	// int16_t delete_count=0;
	// int16_t match_count=0;

	// data from alignment_end input
	uint16_t best_score = a.score;	
	// int32_t ref_len = a.refLen;
	int32_t read_len = a.readLen;
	int32_t ref_end = a.ref;
	int32_t read_end = a.read;
	uint16_t* score = a.scoreArray;
	uint16_t* move = a.moveArray;



	if (best_score ==0)	return 0; // no trace is required. }

	//現在這一步的資訊
	int16_t cur_x, cur_y,  cur_d, cur_score;	
	
	//char op='M', prev_op='M';
	char op, prev_op='-';

	// e: Count the number of M, D or I.
	// l:record length of current cigar
	int32_t e = 0, l=0, s=16;

	while(prev_op == '-' || (cur_score>0 && cur_x>=0 && cur_y >=0)){
		char temp_str[40];
		if (prev_op=='-'){ //抓取出始化位置
			cur_x=ref_end;
			cur_y=read_end;
			cur_d=move[cur_x*read_len+cur_y];	
			cur_score= score[cur_x*read_len+cur_y];	
		}

		//紀錄trace_back的終點
		result->ref_begin = cur_x;
		result->read_begin = cur_y;
		

		sprintf(temp_str, "[%d,%d] %d (%d)",cur_x,cur_y,cur_score,cur_d);
		switch (cur_d){
			case 1 : // D
				cur_x--;
				cur_y--;
				op = 'M';
				++result->match_count; //當然這裡可能不是 macth, 只是放在同一位置
				break;	
			case 2 : // U
				cur_x--;
				op = 'D';
				++result->delete_count;
				break;	
			case 4 : // L
				cur_y--;
				op = 'I';
				++result->insert_count;
				break;	
		}

		if (op == prev_op) 
			++e;
		else {
			if (prev_op !='-'){
				//與前次op不同時，儲存到c, 但若前次為 '-' 則為初始化，暫不處理
				++l;
				while (l >= s) {
					++s;
					kroundup32(s);
					c = (uint32_t*)realloc(c, s * sizeof(uint32_t));
				}
				c[l - 1] = to_cigar_int(e, prev_op);
			}
			//交換狀態與初始化累計次數
			prev_op = op;
			e = 1;
		}
		// 還有下一步，就走下一步
		if (cur_x >=0 && cur_y>=0){
			cur_d=move[cur_x*read_len+cur_y];	
			cur_score= score[cur_x*read_len+cur_y];
		} else{
			break; //沒有下一步，離開迴圈
		}
	}

	// 進行最後部份的輸出
	if (prev_op == 'M') {
		++l;
		while (l >= s) {
			++s;
			kroundup32(s);
			c = (uint32_t*)realloc(c, s * sizeof(uint32_t));
		}
		// 處理最後尚未被紀錄到的部份
		c[l - 1] = to_cigar_int(e, op);
	}else {
		// 這一段是原來的，沒有改，但是我也懷疑最後的比對結果為何會不是'M'
		l += 2;
		while (l >= s) {
			++s;
			kroundup32(s);
			c = (uint32_t*)realloc(c, s * sizeof(uint32_t));
		}
		c[l - 2] = to_cigar_int(e, op);
		c[l - 1] = to_cigar_int(1, 'M');
	}

	// Reverse Sequence
	c1 = (uint32_t*)malloc(l * sizeof(uint32_t));
	s = 0;
	e = l - 1;
	while (LIKELY(s <= e)) {
		c1[s] = c[e];
		c1[e] = c[s];
		++ s;
		-- e;
	}
	result->seq = c1;
	result->length = l;

	free(c);
	return result;

}

s_align* ssw_align (const s_profile* prof,
					const int16_t* ref,
				  	int32_t refLen,
				  	const uint8_t weight_gapO,
				  	const uint8_t weight_gapE,
					const uint8_t flag,	
					//  (from high to low) 
					// bit 5: return the best alignment beginning position;  這個應該一定會回傳。
					// bit 6: if (ref_end1 - ref_begin1 <= filterd) && (read_end1 - read_begin1 <= filterd), return cigar; 
					// bit 7: if max score >= filters, return cigar; 
					// bit 8: always return cigar; 
					// if 6 & 7 are both setted, only return cigar when both filter fulfilled
					const uint16_t filters,
					const int32_t filterd,
					const int32_t maskLen) {

	alignment_end* bests = 0, *bests_reverse = 0;
	__m128i* vP = 0;
	int32_t word = 0, band_width = 0, readLen = prof->readLen;
	int16_t* read_reverse = 0;
	cigar* path=0;
	s_align* r = (s_align*)calloc(1, sizeof(s_align));
	r->ref_begin1 = -1;
	r->read_begin1 = -1;
	r->cigar = 0;
	r->cigarLen = 0;
	if (maskLen < 15) {
		fprintf(stderr, "When maskLen < 15, the function ssw_align doesn't return 2nd best alignment information.\n");
	}

	// 第一輪的 sw_sse 比較
	if (prof->profile_byte) {
		bests = sw_sse2_byte(ref, 0, refLen, readLen, weight_gapO, weight_gapE, prof->profile_byte, -1, prof->bias, maskLen);
		if (prof->profile_word && bests[0].score == 255) {
			free(bests);
			bests = sw_sse2_word(ref, 0, refLen, readLen, weight_gapO, weight_gapE, prof->profile_word, -1, maskLen);
			word = 1;
		} else if (bests[0].score == 255) {
			fprintf(stderr, "Please set 2 to the score_size parameter of the function ssw_init, otherwise the alignment results will be incorrect.\n");
			free(r);
			return NULL;
		}
	}else if (prof->profile_word) {
		bests = sw_sse2_word(ref, 0, refLen, readLen, weight_gapO, weight_gapE, prof->profile_word, -1, maskLen);
		word = 1;
	}else {
		fprintf(stderr, "Please call the function ssw_init before ssw_align.\n");
		free(r);
		return NULL;
	}

	r->score1 = bests[0].score;
	r->ref_end1 = bests[0].ref;
	r->read_end1 = bests[0].read;

	// path_n=back_trace(bests[0]);

	// if (path_n>0){
	// 	r->cigar_n = path_n->seq;
	// 	r->cigarLen_n = path_n->length;
	// 	r->ref_begin2= path_n->ref_begin;
	// 	r->read_begin2 = path_n->read_begin;
	// 	r->insert_count = path_n->insert_count;
	// 	r->delete_count = path_n->delete_count;
	// 	r->match_count= path_n -> match_count;
	// 	free(path_n);
	// }

	// [ADD] 2020/03/11
	// r->n_score = bests[0].n_score;
	r->refLen = bests[0].refLen;
	r->readLen = bests[0].readLen;

	r->scoreArray= bests[0].scoreArray;
	r->moveArray = bests[0].moveArray;
	
	// printf("--------- r --------------\n");
	// for (int i=0; i<r->refLen; ++i){
	// 	for (int j = 0; LIKELY(j < r->readLen ); ++j) {
	// 		// 計算第i,j值，需要讀取哪個位置
	// 		printf("[%d,%d]%d(%d) ",i,j,r->scoreArray_u16[i*r->readLen+j],r->moveArray_u16[i*r->readLen+j]);
	// 	}
	// 	printf("\n");
	// }

	//END


	if (maskLen >= 15) {
		r->score2 = bests[1].score;
		r->ref_end2 = bests[1].ref;
	} else {
		r->score2 = 0;
		r->ref_end2 = -1;
	}
	free(bests);
	// flag ==0, 不回傳cigar
	// flag ==2, 只回傳 score > filter的
	if (flag == 0 || (flag == 2 && r->score1 < filters)) goto end;

	// Find the beginning position of the best alignment.
	read_reverse = seq_reverse(prof->read, r->read_end1);
	if (word == 0) {
		vP = qP_byte(read_reverse, prof->mat, r->read_end1 + 1, prof->n, prof->bias);
		bests_reverse = sw_sse2_byte(ref, 1, r->ref_end1 + 1, r->read_end1 + 1, weight_gapO, weight_gapE, vP, r->score1, prof->bias, maskLen);
	} else {
		vP = qP_word(read_reverse, prof->mat, r->read_end1 + 1, prof->n);
		bests_reverse = sw_sse2_word(ref, 1, r->ref_end1 + 1, r->read_end1 + 1, weight_gapO, weight_gapE, vP, r->score1, maskLen);
	}
	free(vP);
	free(read_reverse);
	r->ref_begin1 = bests_reverse[0].ref;
	r->read_begin1 = r->read_end1 - bests_reverse[0].read;
	free(bests_reverse);

	// flag &7 ==0,  1 or 2 or 4 都沒有set, 那不用回傳
	// flag ==2, flag ==4 的情形下需要再看進一步條件
	// 若可以的話，就回傳

	if ((7&flag) == 0 || ((2&flag) != 0 && r->score1 < filters) || ((4&flag) != 0 && (r->ref_end1 - r->ref_begin1 > filterd || r->read_end1 - r->read_begin1 > filterd))) goto end;

	// Generate cigar. (路徑)
	refLen = r->ref_end1 - r->ref_begin1 + 1;
	readLen = r->read_end1 - r->read_begin1 + 1;
	band_width = abs(refLen - readLen) + 1;
	path = banded_sw(ref + r->ref_begin1, prof->read + r->read_begin1, refLen, readLen, r->score1, weight_gapO, weight_gapE, band_width, prof->mat, prof->n);
	if (path == 0) {
		free(r);
		r = NULL;
	}
	else {
		r->cigar = path->seq;
		r->cigarLen = path->length;
		free(path);
	}

end:
	return r;
}

s_align* ssw_quick_align (const s_profile* prof,
					const int16_t* ref,
				  	int32_t refLen,
				  	const uint8_t weight_gapO,
				  	const uint8_t weight_gapE,
					const uint8_t flag,	
					const uint16_t filters,
					const int32_t filterd) {

	alignment_end* bests = 0;
	// __m128i* vP = 0;
	int32_t  readLen = prof->readLen;
	cigar* path=0;
	s_align* r = (s_align*)calloc(1, sizeof(s_align));
	r->ref_begin1 = -1;
	r->read_begin1 = -1;
	r->cigar = 0;
	r->cigarLen = 0;


	// 第一輪的 sw_sse 比較
	bests = sw_sse2_byte(ref, 0, refLen, readLen, weight_gapO, weight_gapE, prof->profile_byte, -1, prof->bias,-1);
	if (prof->profile_word && bests[0].score == 255) {
		free(bests);
		bests = sw_sse2_word(ref, 0, refLen, readLen, weight_gapO, weight_gapE, prof->profile_word, -1,-1);
		// word = 1;
	}

	r->score1 = bests[0].score;
	r->ref_end1 = bests[0].ref;
	r->read_end1 = bests[0].read;

	path=back_trace(bests[0]);

	if (path>0){
		r->cigar = path->seq;
		r->cigarLen = path->length;
		r->ref_begin1= path->ref_begin;
		r->read_begin1 = path->read_begin;
		r->insert_count = path->insert_count;
		r->delete_count = path->delete_count;
		r->match_count= path -> match_count;
		free(path);
	}

	// [ADD] 2020/03/11
	r->score1 = bests[0].n_score;
	r->refLen = bests[0].refLen;
	r->readLen = bests[0].readLen;

	r->scoreArray= bests[0].scoreArray;
	r->moveArray = bests[0].moveArray;
	
	// printf("--------- r --------------\n");
	// for (int i=0; i<r->refLen; ++i){
	// 	for (int j = 0; LIKELY(j < r->readLen ); ++j) {
	// 		// 計算第i,j值，需要讀取哪個位置
	// 		printf("[%d,%d]%d(%d) ",i,j,r->scoreArray_u16[i*r->readLen+j],r->moveArray_u16[i*r->readLen+j]);
	// 	}
	// 	printf("\n");
	// }

	//END

	free(bests);
	return r;
}


void align_destroy (s_align* a) {
	free(a->cigar);
	free(a->moveArray);
	free(a->scoreArray);
	free(a);
}

uint32_t* add_cigar (uint32_t* new_cigar, int32_t* p, int32_t* s, uint32_t length, char op) {
	if ((*p) >= (*s)) {
		++(*s);
		kroundup32(*s);
		new_cigar = (uint32_t*)realloc(new_cigar, (*s)*sizeof(uint32_t));
	}
	new_cigar[(*p) ++] = to_cigar_int(length, op);
	return new_cigar;
}

uint32_t* store_previous_m (int8_t choice,	// 0: current not M, 1: current match, 2: current mismatch
					   uint32_t* length_m,
					   uint32_t* length_x,
					   int32_t* p,
					   int32_t* s,
					   uint32_t* new_cigar) {

	if ((*length_m) && (choice == 2 || !choice)) {
		new_cigar = add_cigar (new_cigar, p, s, (*length_m), '='); 
		(*length_m) = 0;
	} else if ((*length_x) && (choice == 1 || !choice)) { 
		new_cigar = add_cigar (new_cigar, p, s, (*length_x), 'X'); 
		(*length_x) = 0;
	}
	return new_cigar;
}				

/*! @function:
     1. Calculate the number of mismatches.
     2. Modify the cigar string:
         differentiate matches (=) and mismatches(X); add softclip(S) at the beginning and ending of the original cigar.
    @return:
     The number of mismatches.
	 The cigar and cigarLen are modified.
*/
int32_t mark_mismatch (int32_t ref_begin1,
					   int32_t read_begin1,
					   int32_t read_end1,
					   const int8_t* ref,
					   const int8_t* read,
					   int32_t readLen,
					   uint32_t** cigar,
					   int32_t* cigarLen) {

	int32_t mismatch_length = 0, p = 0, i, length, j, s = *cigarLen + 2;
	uint32_t *new_cigar = (uint32_t*)malloc(s*sizeof(uint32_t)), length_m = 0,  length_x = 0;
	char op;

	ref += ref_begin1;
	read += read_begin1;
	if (read_begin1 > 0) new_cigar[p ++] = to_cigar_int(read_begin1, 'S');
	for (i = 0; i < (*cigarLen); ++i) {
		op = cigar_int_to_op((*cigar)[i]);
		length = cigar_int_to_len((*cigar)[i]);
		if (op == 'M') {
			for (j = 0; j < length; ++j) {
				if (*ref != *read) {
					++ mismatch_length;
					// the previous is match; however the current one is mismatche
					new_cigar = store_previous_m (2, &length_m, &length_x, &p, &s, new_cigar);			
					++ length_x;
				} else {
					// the previous is mismatch; however the current one is matche
					new_cigar = store_previous_m (1, &length_m, &length_x, &p, &s, new_cigar);			
					++ length_m;
				}
				++ ref;
				++ read;
			}
		}else if (op == 'I') {
			read += length;
			mismatch_length += length;
			new_cigar = store_previous_m (0, &length_m, &length_x, &p, &s, new_cigar);			
			new_cigar = add_cigar (new_cigar, &p, &s, length, 'I'); 
		}else if (op == 'D') {
			ref += length;
			mismatch_length += length;
			new_cigar = store_previous_m (0, &length_m, &length_x, &p, &s, new_cigar);			
			new_cigar = add_cigar (new_cigar, &p, &s, length, 'D'); 
		}
	}
	new_cigar = store_previous_m (0, &length_m, &length_x, &p, &s, new_cigar);
	
	length = readLen - read_end1 - 1;
	if (length > 0) new_cigar = add_cigar(new_cigar, &p, &s, length, 'S');
	
	(*cigarLen) = p;	
	free(*cigar);
	(*cigar) = new_cigar;
	return mismatch_length;
}

