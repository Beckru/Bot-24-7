import discord
from discord.ext import commands
from PIL import Image
import io
from codigo import deteccion

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix='-', description="prueba de bot", intents=intents)

@bot.command()
async def ping(ctx):
    await ctx.send('pong')

@bot.command()
async def analisar(ctx):
    if not ctx.message.attachments:
        await ctx.send("Por favor, adjunta una imagen para analizar.")
        return
    
    attachment = ctx.message.attachments[0]
    
    if not attachment.content_type.startswith('image/'):
        await ctx.send("El archivo adjunto no es una imagen válida.")
        return
    
    image_bytes = await attachment.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Usa la función importada para invertir la imagen
    inverted_image = deteccion(image)
    
    with io.BytesIO() as image_binary:
        inverted_image.save(image_binary, 'PNG')
        image_binary.seek(0)
        await ctx.send(file=discord.File(fp=image_binary, filename='deteccion.png'))


@bot.event
async def on_ready():
    print('El bot esta listo')

bot.run('MTI1MTY3MTE3MzI0NzI3MTAzMg.GX67dd.5wBmL8EPei1xGV9Y9RIP0jXcWZFAFmkdI0Sbvo')