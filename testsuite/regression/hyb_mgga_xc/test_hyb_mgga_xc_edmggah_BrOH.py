
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_edmggah_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_edmggah", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.507177571713128e+01, -1.507179889062949e+01, -1.507197241985201e+01, -1.507159197937186e+01, -1.507178732902036e+01, -1.507178732902036e+01, -2.817428811146987e+00, -2.817399658494070e+00, -2.816672975657183e+00, -2.818104813699682e+00, -2.817426203107050e+00, -2.817426203107050e+00, -6.337104807772649e-01, -6.339896662412355e-01, -6.429742270071697e-01, -6.415042593240831e-01, -6.337911413102100e-01, -6.337911413102100e-01, -2.066827129998380e-01, -2.068622658001795e-01, -7.583708241739411e-01, -1.926618859637299e-01, -2.066968863679653e-01, -2.066968863679653e-01, -5.944168319990754e-02, -6.015963115300432e-02, -1.111850685491397e-01, -5.211212848727661e-02, -5.962871766442990e-02, -5.962871766442990e-02, -7.553060878175184e+00, -5.227931628424360e+00, -2.316927459577217e+00, -8.263235875994893e+00, -3.388422719879075e+00, -4.683415321767834e+00, 4.214125210055006e-01, -2.118651783861726e+00, -1.839594255388548e+00, -1.854150774008812e+00, -1.843604732491104e+00, -1.634067241881324e+00, -1.185161030854874e+00, -8.645736535198691e-01, -9.918784616976496e-01, -5.019902289025522e-01, -5.422896315982610e-01, -5.796712897799240e-01, -3.746645554675063e-02, -4.008203016863242e-01, 9.878758725479055e-02, -1.622463845963095e+00, -1.593472682805629e-01, -1.375292725364623e-01, -1.014303314879803e-01, -1.672769938750037e+00, -3.004445940944723e-03, -3.209545024992206e-02, -7.015526404710701e-02, -3.649795285136454e-03, -8.586785218049802e+02, 2.978575332754315e+02, 5.666822252006928e+03, -1.175327539248690e+01, 2.319162303574300e+02, -1.082185308118235e+04, 4.103744025188495e+03, -4.558536153273740e-01, -8.837387516554578e-01, -3.672767721866230e-01, -5.192167636462907e-01, -1.183526854303200e+00, -1.316544418382345e+00, 4.573339258682660e-01, 9.801554139614195e-01, -8.692258693661313e-02, -3.939572779321748e-01, -3.483974340107746e-01, -4.226542290356826e-01, -1.938687373006007e-01, -1.369467942377566e-01, -3.066875759701101e-01, 1.412164402881967e-02, -1.306544638383070e-01, -8.825799047556445e-03, -1.671223017015130e-01, -2.086962924595246e-03, -4.897589388332719e-02, -3.103783751643351e-03, -3.103739628517796e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_edmggah_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_edmggah", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.279567487339378e+01, -2.279574590298934e+01, -2.279608674610227e+01, -2.279492322699572e+01, -2.279571212572734e+01, -2.279571212572734e+01, -3.632360877482050e+00, -3.632407047002361e+00, -3.633841598659841e+00, -3.632356457236563e+00, -3.632399072142575e+00, -3.632399072142575e+00, -6.996324141449130e-01, -6.985658462959157e-01, -6.738233720584963e-01, -6.800358718627750e-01, -6.992462476079047e-01, -6.992462476079047e-01, -2.095668323722174e-01, -2.109488371585659e-01, -8.298109808983593e-01, -1.785844211055110e-01, -2.099272553350761e-01, -2.099272553350761e-01, -3.226874632611534e-02, -3.292219902779472e-02, -8.028092443172805e-02, -2.436074232841085e-02, -3.265199724947211e-02, -3.265199724947212e-02, -6.861984721798780e+00, -5.539030648594810e+00, -4.480583847237208e+00, -7.307314556883124e+00, -5.269741244310040e+00, -5.437851837306201e+00, -2.512258911123826e+00, -2.099290855836566e+00, -1.956097090532911e+00, -1.970842143439284e+00, -1.984218744172086e+00, -1.976642967060214e+00, -8.704420253964242e-01, -7.777996365628327e-01, -7.819178284419029e-01, -6.453377054301472e-01, -6.645743774237157e-01, -6.609688060058879e-01, -1.045497710744372e-01, -3.372924881067654e-01, -6.268559760527570e-02, -2.074045048236740e+00, -1.461616384969510e-01, -1.170721035414152e-01, -6.229510612477566e-02, -9.895928257473792e-01, -4.002562844916136e-03, -5.907311588375445e-02, -4.147326066989521e-02, -4.861455122157125e-03, 3.622717305120219e+02, -1.340528922758117e+02, -2.540536257273603e+03, 1.111131290395016e+00, -1.046366241598721e+02, 4.784719752393677e+03, -1.817555649823721e+03, -5.506052565214639e-01, -7.289624323344750e-01, -5.781012768668550e-01, -5.917047891321416e-01, -8.304970000794138e-01, -9.721435893208945e-01, -3.486901728019526e-01, -5.550350269929888e-01, -3.600975854883904e-01, -3.819356701273088e-01, -3.573612469697147e-01, -4.963358585539949e-01, -1.502559336148154e-01, -1.099791860711331e-01, -3.713722849820357e-01, -5.706200981073812e-02, -1.100226228904437e-01, -1.173932818401595e-02, -1.002842354744760e-01, -2.780983834873682e-03, -8.038615871344891e-02, -4.134843302769180e-03, -4.134733959971048e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_edmggah_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_edmggah", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.048077416706066e-09, -5.048013458970401e-09, -5.047741775421065e-09, -5.048789384030057e-09, -5.048043547455407e-09, -5.048043547455407e-09, -6.829017979741969e-06, -6.829007265939091e-06, -6.827643640465869e-06, -6.824478943033952e-06, -6.828901348531223e-06, -6.828901348531223e-06, -3.136806566247344e-03, -3.135225494246773e-03, -2.990186438400139e-03, -2.964043133969884e-03, -3.136504319097331e-03, -3.136504319097331e-03, -2.672449722875377e-01, -2.730882562410138e-01, -1.744669472394158e-03, -2.598453894237229e-01, -2.695832649743218e-01, -2.695832649743218e-01, -1.519868288083205e+03, -1.338508727029782e+03, -1.799467798476574e+01, -1.097857620642545e+04, -1.391977795782886e+03, -1.391977795782886e+03, -2.621342978018994e-07, -9.019915358829699e-07, 1.235991465219975e-07, -1.848577312944143e-07, -6.402222255710791e-07, -1.201683854904321e-06, 1.433131928680695e-05, -2.930171908879338e-05, -4.739242493064148e-05, -4.607189796988530e-05, -4.676428263905997e-05, -5.965266079454658e-05, 2.231781289225816e-03, 5.755994626962515e-05, 2.288373144543292e-03, -6.347336467326731e-03, -4.929219356999479e-03, -4.061771726758361e-03, 1.945751370392432e+00, 2.003239189181060e-01, 2.003996202166773e+00, -6.615011952189246e-05, -7.895699065509774e-01, -1.821348263153862e+00, -4.584401078408745e+03, -8.483779676178692e+01, -2.652696276343816e-09, 4.082577179483827e+00, -6.983669897202564e+03, -5.286013837762668e-05, 3.274103292874850e-03, 3.360128527530456e-03, 3.332187501030489e-03, 3.187097706706811e-03, 3.321494674695720e-03, 3.319357887760313e-03, 3.781828845034006e-03, -9.484686070022941e-03, 2.437373582805459e-03, -1.234066501480547e-04, -6.125519784660649e-03, 3.479618514654824e-03, 1.188285736070399e-03, 1.494517382857448e-01, 7.945264339234440e-02, 3.061521260409614e-02, -4.497230889667081e-03, -2.423660770854313e-02, -1.346006178446023e-02, -5.010622937129890e+00, -5.201095673649407e+00, -5.062306153983209e-02, 2.916139887712261e+00, -2.983558043688481e+00, 3.463044702137788e-03, -5.315521756680367e+04, -1.479583338107709e-04, 2.355841652612377e+00, -5.086988694406207e-02, -8.173571481058841e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_edmggah_BrOH_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_edmggah", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-2.006843709691347e-04, -2.006832929537950e-04, -2.006800381560827e-04, -2.006976797785746e-04, -2.006837884602249e-04, -2.006837884602249e-04, -1.325082039382856e-03, -1.325086697358300e-03, -1.325162746127211e-03, -1.324762394448836e-03, -1.325078978164627e-03, -1.325078978164627e-03, -6.703273559829815e-03, -6.694432520899575e-03, -6.368051906975794e-03, -6.381999392518173e-03, -6.700366028032428e-03, -6.700366028032428e-03, -1.656951624750901e-02, -1.693594751877555e-02, -5.850929603434428e-03, -1.270286004538611e-02, -1.669194365730337e-02, -1.669194365730337e-02, -6.370528099685456e-03, -6.447555787142535e-03, -7.586776177765956e-03, -5.080578788180233e-03, -6.497668476077917e-03, -6.497668476077917e-03, -2.332614839547246e-04, -5.913766009832638e-04, -1.751185332662762e-05, -1.900630424827230e-04, -4.448100933642543e-04, -7.589235413696117e-04, -1.062905496905067e-04, -1.411579979566920e-03, -1.926629476405916e-03, -1.905166929284843e-03, -1.944882729229778e-03, -2.336695279302377e-03, -1.915509079361775e-03, -3.203153410795624e-03, -2.390139584077108e-03, -8.468499744339695e-03, -7.746825395992003e-03, -6.993591399710086e-03, -1.782442652872214e-03, -5.194371022546478e-03, -1.752830929226884e-03, -2.399807347903635e-03, -1.418540736312268e-02, -1.998617831806832e-02, -1.904623106169156e-03, -5.260999801674369e-05, -6.926127485311046e-16, -8.995417131462193e-04, -3.275966601163569e-03, -2.479614145509121e-11, -7.067385358281641e-04, -7.043146814781947e-04, -7.024973670841208e-04, -8.058991601048749e-04, -7.017988503531496e-04, -7.036047778060140e-04, -7.304435521935144e-04, -9.168817900512851e-03, -2.750289507153346e-03, -4.029649404380629e-03, -7.780723033704239e-03, -1.932002586813000e-03, -1.691727446568269e-03, -1.827571327708317e-03, -1.627499195789852e-03, -1.320472278238513e-03, -7.889170055123707e-03, -1.022721754695599e-02, -9.949352851352761e-03, -2.617397661780312e-03, -6.806951291807486e-03, -1.423867711911308e-02, -1.418057882142011e-03, -1.288885814637707e-02, -1.102184099012269e-08, -6.764611961033912e-04, -1.290765706101416e-11, -2.658321310545882e-03, -1.464773443313067e-08, -2.353539769986746e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_edmggah_BrOH_1_vtau():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_edmggah", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [7.895175245881120e-04, 7.895132759900840e-04, 7.895005866540834e-04, 7.895701129558987e-04, 7.895152276302527e-04, 7.895152276302527e-04, 4.854616639244655e-03, 4.854637018554150e-03, 4.855023506659790e-03, 4.853456754136832e-03, 4.854608839942693e-03, 4.854608839942693e-03, 1.957204473754726e-02, 1.952623790422836e-02, 1.794225394078024e-02, 1.807974003328084e-02, 1.955660842141669e-02, 1.955660842141669e-02, 4.194264116829555e-02, 4.350465067901192e-02, 1.783225516357602e-02, 2.604231254220413e-02, 4.246080723858577e-02, 4.246080723858578e-02, 2.548210706934721e-02, 2.579021069614410e-02, 2.610805436806127e-02, 2.032231515272093e-02, 2.599066358744952e-02, 2.599066358744952e-02, 7.250896117094085e-04, 2.157620725967193e-03, -1.379014553581321e-04, 5.523582295013541e-04, 1.571319960908933e-03, 2.827773752930362e-03, -8.494776770494706e-04, 4.387450782060856e-03, 6.425418687583103e-03, 6.351995633668252e-03, 6.523785407051373e-03, 8.091035607341767e-03, -1.088816210361881e-03, 4.987740610208376e-03, 1.853672638575119e-04, 2.500252095197649e-02, 2.266741791979002e-02, 1.965448193466235e-02, -1.412828130086446e-02, -3.506631927227860e-03, -1.402007171395669e-02, 8.280146236421605e-03, 3.335036821497929e-02, 5.655345203476184e-02, 7.618492424676622e-03, 2.104399920667490e-04, 2.770449874744145e-15, -7.146231191498548e-03, 1.310386640465427e-02, 9.918455936345909e-11, -5.564189395147537e-03, -5.634517423711039e-03, -5.619978936595121e-03, -5.189602472651078e-03, -5.614390756438624e-03, -5.607167046627166e-03, -5.843548417401091e-03, 2.615113731330273e-02, 1.037970480368445e-03, 6.638693428098353e-03, 2.140097868827179e-02, -1.993903099293161e-03, -5.605466569460534e-04, -1.461777951901486e-02, -1.301899720184474e-02, -1.046594920057841e-02, 1.390110021669551e-02, 2.325329018402465e-02, 2.787353498361993e-02, 7.278139142286333e-03, 1.884971113285305e-02, 4.032558625579588e-02, -1.133877011983707e-02, 3.454443093710319e-02, -3.844566531245577e-08, 2.705844784413565e-03, 5.163062824405665e-11, -5.292115674036976e-03, 5.859093773251729e-08, 9.414159079409123e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05